/**
 * Benchmark comparison of Cholesky decomposition between CUDA and OpenCL backends.
 */
#include <bandicoot>
#include <armadillo>

using namespace coot;



template<typename MatType>
double run_benchmark(const uword rows, const bool cuda, const bool lup)
  {
  // set the correct backend
  if (cuda)
    get_rt().backend = CUDA_BACKEND;
  else
    get_rt().backend = CL_BACKEND;

  MatType x;
  x.set_size(rows, rows);
  x.randu();
  x.diag() += 0.5;

  MatType l, u, p;

  // finish all delayed operations
  get_rt().synchronise();

  wall_clock c;
  double t;

  c.tic();
  bool status;
  if (lup)
    status = lu(l, u, p, x);
  else
    status = lu(l, u, x);
  get_rt().synchronise();
  t = c.toc();

  if (!status)
    std::cerr << "lu() failed!\n";

  // sanity check
  MatType y;
  if (lup)
    y = p.t() * l * u;
  else
    y = l * u;
  y -= x;
  const typename MatType::elem_type diff = accu(square(y)) / y.n_elem;
  if (diff > 1e-5)
    std::cerr << "reconstruction diff too high: " << diff << "\n";

  return t;
  }



template<typename MatType>
void run_benchmarks(const uword rows,
                    const bool cuda,
                    const bool lup,
                    const size_t trials,
                    const char* task_name,
                    const char* device_name,
                    const char* backend_name,
                    const char* elem_type,
                    std::ofstream& out)
  {
  for (size_t trial = 0; trial < trials; ++trial)
    {
    const double t = run_benchmark<MatType>(rows, cuda, lup);

    out << task_name << "," << device_name << "," << backend_name << "," << elem_type << ","
        << rows << "," << rows << "," << trial << "," << t << "\n";
    std::cout << task_name << ", " << device_name << ", " << backend_name << ", " << elem_type << ", "
        << rows << ", " << rows << ", " << trial << ", " << t << "\n";
    }
  }



int main(int argc, char** argv)
  {
  if (argc != 5)
    {
    std::cerr << "Usage: " << argv[0] << " device_name trials rows out_csv" << std::endl;
    exit(1);
    }

  const char* device_name = argv[1];
  const size_t trials = (size_t) atoi(argv[2]);
  const uword rows = (uword) atoi(argv[3]);
  const char* out_csv = argv[4];

  wall_clock c;

  std::cout << "lu: LU decomposition benchmark comparison\n";
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

  run_benchmarks<arma::fmat>(rows, false, false, trials, "lu", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, false, false, trials, "lu", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, true, false, trials, "lu", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::mat>(rows, false, false, trials, "lu", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, false, false, trials, "lu", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, true, false, trials, "lu", device_name, "cuda", "double", out_file);

  run_benchmarks<arma::fmat>(rows, false, true, trials, "lup", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, false, true, trials, "lup", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, true, true, trials, "lup", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::mat>(rows, false, true, trials, "lup", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, false, true, trials, "lup", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, true, true, trials, "lup", device_name, "cuda", "double", out_file);
  }
