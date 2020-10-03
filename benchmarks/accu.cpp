/**
 * Benchmark comparison of accu() between CUDA and OpenCL backends.
 */
#include <bandicoot>
#include <armadillo>

using namespace coot;



template<typename MatType>
void fill_randu(MatType& x)
  {
  arma::Mat<typename MatType::elem_type> cpu_x(x.n_rows, x.n_cols);

  fill_randu(cpu_x);

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



template<>
void fill_randu(arma::Mat<s32>& x)
  {
  x = arma::randi<arma::Mat<s32>>(x.n_rows, x.n_cols, arma::distr_param(0, 10));
  }



template<>
void fill_randu(arma::Mat<s64>& x)
  {
  x = arma::randi<arma::Mat<s64>>(x.n_rows, x.n_cols, arma::distr_param(0, 10));
  }



template<typename MatType>
double run_benchmark(const uword elem, const bool cuda)
  {
  // set the correct backend
  if (cuda)
    get_rt().backend = CUDA_BACKEND;
  else
    get_rt().backend = CL_BACKEND;

  MatType x;
  x.set_size(elem, 1);
  fill_randu(x);

  // finish all delayed operations
  get_rt().synchronise();

  wall_clock c;
  double t;

  c.tic();
  const double result = accu(x);
  get_rt().synchronise();
  t = c.toc();

  // prevent optimization of accu() call
  if (result < 0)
    std::cerr << "something wrong with the results!\n";

  return t;
  }



template<typename MatType>
void run_benchmarks(const uword elem,
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
    const double t = run_benchmark<MatType>(elem, cuda);
    const double bw = (elem * sizeof(typename MatType::elem_type) / t) / std::pow(2.0, 30.0);

    out << task_name << "," << device_name << "," << backend_name << "," << elem_type << ","
        << elem << ",1," << trial << "," << t << "," << bw << "\n";
    std::cout << task_name << ", " << device_name << ", " << backend_name << ", " << elem_type << ", "
        << elem << ", 1, " << trial << ", " << t << ", " << bw << "\n";
    }
  }



int main(int argc, char** argv)
  {
  if (argc != 5)
    {
    std::cerr << "Usage: " << argv[0] << " device_name trials elem out_csv" << std::endl;
    exit(1);
    }

  const char* device_name = argv[1];
  const size_t trials = (size_t) atoi(argv[2]);
  const uword elem = (uword) atoi(argv[3]);
  const char* out_csv = argv[4];

  wall_clock c;

  std::cout << "accu: element accumulation benchmark comparison\n";
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

  run_benchmarks<arma::Mat<s32>>(elem, false, trials, "accu", device_name, "cpu", "int32", out_file);
  run_benchmarks<coot::Mat<s32>>(elem, false, trials, "accu", device_name, "opencl", "int32", out_file);
  run_benchmarks<coot::Mat<s32>>(elem, true, trials, "accu", device_name, "cuda", "int32", out_file);

  run_benchmarks<arma::Mat<s64>>(elem, false, trials, "accu", device_name, "cpu", "int64", out_file);
  run_benchmarks<coot::Mat<s64>>(elem, false, trials, "accu", device_name, "opencl", "int64", out_file);
  run_benchmarks<coot::Mat<s64>>(elem, true, trials, "accu", device_name, "cuda", "int64", out_file);

  run_benchmarks<arma::fmat>(elem, false, trials, "accu", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(elem, false, trials, "accu", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(elem, true, trials, "accu", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::mat>(elem, false, trials, "accu", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(elem, false, trials, "accu", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(elem, true, trials, "accu", device_name, "cuda", "double", out_file);
  }
