#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/io/matrix_market.h>
#include <cusp/precond/diagonal.h>

#include <fstream>
#include <iostream>
#include <sys/time.h>

void load_vector(const char* flname, cusp::array1d<float, cusp::host_memory> &v)
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

void write_vector(const char* flname, cusp::array1d<float, cusp::host_memory> &v)
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

  
  // create an empty sparse matrix structure (CSR format)
  cusp::csr_matrix<int, float, cusp::host_memory> A_host;
  // read matrix
  cusp::io::read_matrix_market_file(A_host, argv[2]);

  // create empty array
  cusp::array1d<float, cusp::host_memory> b_host(A_host.num_cols, 0);
  // read vector
  load_vector(argv[3], b_host);

  std::cout << "Matrix dimensions: " << A_host.num_rows << " " << A_host.num_cols << std::endl;
  std::cout << "Vector length : " << b_host.size() << std::endl;      

  struct timespec copy_start;
  struct timespec copy_end;
  struct timespec exec_start;
  struct timespec exec_end;

  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &copy_start);
  cusp::csr_matrix<int, float, cusp::device_memory> A(A_host);
  cusp::array1d<float, cusp::device_memory> b(b_host);  
  cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &copy_end);
  
  // set stopping criteria:
  //  iteration_limit    = 5000
  //  relative_tolerance = 1e-6
  //  absolute_tolerance = 1e-6
  //  verbose            = false
  cusp::monitor<float> monitor(b, 5000, 1e-6, 1e-6, false);
  
  // solve the linear system A x = b
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &exec_start);
  if (strcmp(argv[1], PRECOND_DIAG) == 0)
    {
      cusp::precond::diagonal<float, cusp::device_memory> M(A);
      cusp::krylov::cg(A, x, b, monitor, M);
    }
    else if(strcmp(argv[1], PRECOND_NONE) == 0)
    {
      cusp::identity_operator<float, cusp::device_memory> M(A.num_rows, A.num_cols);
      cusp::krylov::cg(A, x, b, monitor, M);
    }
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &exec_end);

  // copy results back and write out
  cusp::array1d<float, cusp::host_memory> x_host(x);
  write_vector(argv[4], x_host);

  if(monitor.converged())
    {
      std::cout << "Solver converged to " << monitor.relative_tolerance() << " relative tolerance";
      std::cout << " with residual norm " << monitor.residual_norm();
      std::cout << " after " << monitor.iteration_count() << " iterations" << std::endl;
    } else {
      std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
      std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " << std::endl;
    }

  long int copy_time = elapsed_time_ms(copy_start, copy_end);
  long int execution_time = elapsed_time_ms(exec_start, exec_end);
  std::cout << "Copy time (ms): " << copy_time << std::endl;
  std::cout << "Execution time (ms): " << execution_time << std::endl;
  
  return 0;
}
