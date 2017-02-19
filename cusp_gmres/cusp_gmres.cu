#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/gmres.h>
#include <cusp/gallery/poisson.h>

#include <fstream>
#include <stdio>

void load_vector(const char* flname, cusp::array1d<float, cusp::host_memory>& v)
{
  std::ifstream input_data;
  input_data.open(flname);

  int n_elements;
  input_data >> n_elements;

  for(int i = 0; i < n_elements; i++)
    {
      float entry;
      input_data >> n_elements;
      n_elements.push_back(entry);
    }

  input_data.close();
}

void write_vector(const char* flname, cusp::array1d<float, cusp::host_memory>& v)
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


int main(int argc, char** argv)
{
  if(argc != 4)
    {
      std::cout << "Usage: " << argv[0] << " <matrix_flname> <input_vector_flname> <output_vector_flname>" << std::endl;
    }
  
  // create an empty sparse matrix structure (CSR format)
  cusp::csr_matrix<int, float, cusp::device_memory> A;
  // read matrix
  cusp::io::read_matrix_market_file(A, argv[1]);

  // create empty array
  cusp::array1d<float, cusp::host_memory> b_host(A.num_rows, 0);
  // read vector
  load_vector(argv[2], &b_host);

  cusp::array1d<float, cusp::device_memory> b(b_host);  
  cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);

  // set stopping criteria:
  //  iteration_limit    = 5000
  //  relative_tolerance = 1e-6
  //  absolute_tolerance = 0
  //  verbose            = true
  cusp::monitor<float> monitor(b, 5000, 1e-6, 0, true);
  int restart = 50;
  // set preconditioner (identity)
  cusp::identity_operator<float, cusp::device_memory> M(A.num_rows, A.num_rows);
  // solve the linear system A x = b
  cusp::krylov::gmres(A, x, b,restart, monitor, M);

  cusp::array1d<float, cusp::host_memory> x_host(x);
  write_vector(argv[3], &x_host);
  
  return 0;
}
