/**
 * Benchmark comparison of matrix copies between CUDA and OpenCL backends.
 */
#include <bandicoot>
#include <armadillo>

using namespace coot;


template<typename MatType>
void fill_randu(MatType& x)
  {
  arma::Mat<typename MatType::elem_type> cpu_x(x.n_rows, x.n_cols);

  cpu_x.randu();

  x.copy_into_dev_mem(cpu_x.memptr(), x.n_elem);
  }



template<>
void fill_randu(arma::fmat& x)
  {
  x.randu();
  }



template<>
void fill_randu(arma::mat& x)
  {
  x.randu();
  }



template<typename MatType>
double run_benchmark(const uword rows, const uword cols, const bool cuda)
  {
  // set the correct backend
  if (cuda)
    get_rt().backend = CUDA_BACKEND;
  else
    get_rt().backend = CL_BACKEND;

  MatType x;
  x.set_size(rows, cols);
  fill_randu(x); // start with random memory

  MatType y;
  y.set_size(rows, cols);

  // finish all delayed operations
  get_rt().synchronise();

  wall_clock c;
  double t;

  c.tic();
  y = x;
  get_rt().synchronise();
  t = c.toc();

  return t;
  }


template<typename MatType>
void run_benchmarks(const uword rows,
                    const uword cols,
                    const bool cuda,
                    const size_t trials,
                    const char* task_name,
                    const char* device_name,
                    const char* backend_name,
                    const char* elem_type,
                    std::ofstream& out)
  {
  for (size_t trial = 0; trial < trials; ++trial)
    {
    const double t = run_benchmark<MatType>(rows, cols, cuda);

    out << task_name << "," << device_name << "," << backend_name << "," << elem_type << ","
        << rows << "," << cols << "," << trial << "," << t << "\n";
    std::cout << task_name << ", " << device_name << ", " << backend_name << ", " << elem_type << ", "
        << rows << ", " << cols << ", " << trial << ", " << t << "\n";
    }
  }



int main(int argc, char** argv)
  {
  if (argc != 6)
    {
    std::cerr << "Usage: " << argv[0] << " device_name trials rows cols out_csv" << std::endl;
    exit(1);
    }

  const char* device_name = argv[1];
  const size_t trials = (size_t) atoi(argv[2]);
  const uword rows = (uword) atoi(argv[3]);
  const uword cols = (uword) atoi(argv[4]);
  const char* out_csv = argv[5];

  wall_clock c;

  std::cout << "fill: matrix fill with zeros benchmark comparison\n";
  std::cout << "  bandicoot version " << coot::coot_version::as_string() << '\n';
  std::cout << "  armadillo version " << arma::arma_version::as_string() << '\n';
  std::cout << '\n';

  // Time initialization.
  c.tic();
  coot::get_rt().init(true);
  coot::get_rt().cuda_rt.init(false, 0, 0, true);
  double time = c.toc();
  std::cout << "bandicoot initialization time: " << time << "s\n";

  std::ofstream out_file(out_csv, std::ios_base::app);
  if (!out_file.is_open())
    {
    std::cerr << "failed to open " << out_csv << "!\n";
    exit(1);
    }

  run_benchmarks<arma::fmat>(rows, cols, false, trials, "copy", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, cols, false, trials, "copy", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, cols, true, trials, "copy", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::mat>(rows, cols, false, trials, "copy", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, cols, false, trials, "copy", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, cols, true, trials, "copy", device_name, "cuda", "double", out_file);
  }
