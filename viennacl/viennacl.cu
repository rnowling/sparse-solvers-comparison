#include <fstream>
#include <iostream>
#include <sys/time.h>

#include <map>
#include <vector>

#ifndef NDEBUG
#define BOOST_UBLAS_NDEBUG
#endif

#include <boost/numeric/ublas/matrix_sparse.hpp>

#ifndef VIENNACL_WITH_CUDA
#define VIENNACL_WITH_CUDA
#endif

#define VIENNACL_WITH_UBLAS 1 

#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/row_scaling.hpp>

using namespace boost::numeric;

void load_vector(const char* flname, std::vector<float> &v)
{
  std::ifstream input_data;
  input_data.open(flname);

  int n_elements;
  input_data >> n_elements;

  v.resize(n_elements);

  for(int i = 0; i < n_elements; i++)
    {
      float entry;
      input_data >> entry;
      v[i] = entry;
    }

  input_data.close();
}

void load_matrix(const char* filename, ublas::compressed_matrix<float> &cpu_sparse_matrix,
                 unsigned int &n_rows, unsigned int &n_cols)
{
  unsigned n_nnzs;
  char buffer[255];
  FILE* matrix_file = fopen(filename, "r");
  
  // skip comments
  do {
    fgets(buffer, 255, matrix_file);
  } while(buffer[0] == '%');
  
  // read matrix dimensions
  if (sscanf(buffer, "%u %u %u", &n_rows, &n_cols, &n_nnzs) != 3)
    {
      printf("Failed to parse matrix entry\n");
      printf("Line: '%s'\n", buffer);
      fclose(matrix_file);
      exit(1);
    }
  
  printf("Reading a %d x %d matrix\n", n_rows, n_cols);

  cpu_sparse_matrix.resize(n_rows, n_cols, false);
    
  // read in data
  printf("Reading data\n");
  while(fgets(buffer, 255, matrix_file) != NULL && !feof(matrix_file))
    {
      unsigned int row, col;
      float value;
      
      if (sscanf(buffer, "%u %u %f", &row, &col, &value) != 3)
        {
          printf("Failed to parse matrix entry\n");
          printf("Line: '%s'\n", buffer);
          fclose(matrix_file);
          exit(1);
        }
      
      // convert from 1-indexing to 0-indexing
      cpu_sparse_matrix(row - 1, col - 1) = value;
    }
  
  fclose(matrix_file);
    
  printf("Read data\n");
}

void write_vector(const char* flname, std::vector<float> &v)
{
  std::ofstream output_data;
  output_data.open(flname, std::ofstream::out | std::ofstream::trunc);

  output_data << v.size() << std::endl;
  for(int i = 0; i < v.size(); i++)
    {
      output_data << v[i] << std::endl;
    }
  
  output_data.close();  
}

long int elapsed_time_ms(struct timespec &start, struct timespec &end)
{
  return (end.tv_sec * 1000 + end.tv_nsec / (1000 * 1000)) -
    (start.tv_sec * 1000 + start.tv_nsec / (1000 * 1000));
}


int main(int argc, char** argv)
{
  const char* PRECOND_NONE = "none";
  const char* PRECOND_DIAG = "diag";
  
  if(argc != 5)
    {
      std::cout << "Usage: " << argv[0] << " <preconditioner> <matrix_flname> <input_vector_flname> <output_vector_flname>" << std::endl;
      std::cout << std::endl;
      std::cout << "Preconditioner can be one of: " << PRECOND_NONE << " " << PRECOND_DIAG << std::endl;
      return 1;
    }

  if(strcmp(argv[1], PRECOND_NONE) != 0 and strcmp(argv[1], PRECOND_DIAG) != 0)
    {
      std::cout << "Preconditioner must be one of: " << PRECOND_NONE << " " << PRECOND_DIAG << std::endl;
      return 1;
    }
  else
    {
      std::cout << "Using preconditioner: " << argv[1] << std::endl;
    }

  unsigned int n_rows, n_cols;
  ublas::compressed_matrix<float> A_host;
  load_matrix(argv[2], A_host, n_rows, n_cols);
  viennacl::compressed_matrix<float> A_gpu(n_rows, n_cols);

  std::vector<float> b_host;
  load_vector(argv[3], b_host);
  viennacl::vector<float> b_gpu(b_host.size());

  std::cout << "Read dimensions: " << n_rows << " " << n_cols << std::endl;
  std::cout << "CPU Matrix dimensions: " << A_host.size1() << " " << A_host.size2() << std::endl;
  std::cout << "GPU Matrix dimensions: " << A_gpu.size1() << " " << A_gpu.size2() << std::endl;
  std::cout << "Vector length : " << b_host.size() << std::endl;      

  struct timespec copy_start;
  struct timespec copy_end;
  struct timespec exec_start;
  struct timespec exec_end;

  std::cout << "Copying data" << std::endl;
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &copy_start);
  copy(b_host.begin(), b_host.end(), b_gpu.begin());
  copy(A_host, A_gpu);
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &copy_end);
  
  // set stopping criteria:
  //  iteration_limit    = 5000
  //  relative_tolerance = 1e-6
  //  absolute_tolerance = 1e-6
  //  verbose            = true
  viennacl::linalg::gmres_tag my_tag(1e-6, 5000, 50);
  
  // solve the linear system A x = b
  std::cout << "Solve system" << std::endl;
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &exec_start);
  viennacl::vector<float> x_gpu;
  // set preconditioner
  if (strcmp(argv[1], PRECOND_DIAG) == 0)
    {
      viennacl::linalg::row_scaling< viennacl::compressed_matrix<float> > vcl_row_scaling(A_gpu, viennacl::linalg::row_scaling_tag());
      x_gpu = viennacl::linalg::solve(A_gpu, b_gpu, my_tag, vcl_row_scaling);
    }
  else if(strcmp(argv[1], PRECOND_NONE) == 0)
    {
      x_gpu = viennacl::linalg::solve(A_gpu, b_gpu, my_tag);
    }
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &exec_end);

  // copy results back and write out
  std::vector<float> x_host(n_cols);
  copy(x_gpu.begin(), x_gpu.end(), x_host.begin());
  write_vector(argv[4], x_host);

  std::cout << "Solver converged to " << my_tag.error() << " relative tolerance";
  std::cout << " after " << my_tag.iters() << " iterations" << std::endl;

  long int copy_time = elapsed_time_ms(copy_start, copy_end);
  long int execution_time = elapsed_time_ms(exec_start, exec_end);
  std::cout << "Copy time (ms): " << copy_time << std::endl;
  std::cout << "Execution time (ms): " << execution_time << std::endl;
  
  return 0;
}
