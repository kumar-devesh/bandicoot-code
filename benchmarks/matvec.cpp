/**
 * Benchmark comparison of matrix-vector multiplication between CUDA and OpenCL backends.
 */
#include <bandicoot>
#include <armadillo>

using namespace coot;



template<typename MatType>
void fill_randu(MatType& x, MatType& y)
  {
  arma::Mat<typename MatType::elem_type> cpu_x, cpu_y, cpu_z;
  cpu_x.set_size(x.n_rows, x.n_cols);
  cpu_y.set_size(y.n_rows, y.n_cols);

  cpu_x.randu();
  cpu_y.randu();

  x.copy_into_dev_mem(cpu_x.memptr(), x.n_elem);
  y.copy_into_dev_mem(cpu_y.memptr(), y.n_elem);
  }



template<>
void fill_randu(arma::fmat& x, arma::fmat& y)
  {
  x.randu();
  y.randu();
  }



template<>
void fill_randu(arma::mat& x, arma::mat& y)
  {
  x.randu();
  y.randu();
  }



template<typename MatType>
double run_benchmark(const uword rows, const bool trans_a, const bool trans_b, const bool cuda)
  {
  // set the correct backend
  if (cuda)
    get_rt().backend = CUDA_BACKEND;
  else
    get_rt().backend = CL_BACKEND;

  MatType x;
  x.set_size(rows, rows);

  MatType y;
  if (trans_b)
    y.set_size(1, rows);
  else
    y.set_size(rows, 1);

  fill_randu(x, y);

  MatType z(rows, 1);

  // finish all delayed operations
  get_rt().synchronise();

  wall_clock c;
  double t;
  if (trans_a && trans_b)
    {
    c.tic();
    z = x.t() * y.t();
    get_rt().synchronise();
    t = c.toc();
    }
  else if (trans_a)
    {
    c.tic();
    z = x.t() * y;
    get_rt().synchronise();
    t = c.toc();
    }
  else if (trans_b)
    {
    c.tic();
    z = x * y.t();
    get_rt().synchronise();
    t = c.toc();
    }
  else
    {
    c.tic();
    z = x * y;
    get_rt().synchronise();
    t = c.toc();
    }

  return t;
  }



template<typename MatType>
void run_benchmarks(const uword rows,
                    const bool trans_a,
                    const bool trans_b,
                    const bool cuda_backend,
                    const size_t trials,
                    const char* task_name,
                    const char* device_name,
                    const char* backend_name,
                    const char* elem_type,
                    std::ofstream& out)
  {
  for (size_t trial = 0; trial < trials; ++trial)
    {
    const double t = run_benchmark<MatType>(rows, trans_a, trans_b, cuda_backend);

    out << task_name << "," << device_name << "," << backend_name << "," << elem_type << ","
        << rows << ",1," << trial << "," << t << "\n";
    std::cout << task_name << ", " << device_name << ", " << backend_name << ", " << elem_type << ", "
        << rows << ", 1, " << trial << ", " << t << "\n";
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

  std::cout << "matvec: Matrix-vector multiplication benchmark comparison\n";
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

  // Run benchmarks for all four combinations of transposition and all the backends.
  run_benchmarks<arma::fmat>(rows, false, false, false, trials, "matvec", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, false, false, false, trials, "matvec", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, false, false, true, trials, "matvec", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::fmat>(rows, true, false, false, trials, "matvec-at", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, true, false, false, trials, "matvec-at", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, true, false, true, trials, "matvec-at", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::fmat>(rows, false, true, false, trials, "matvec-bt", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, false, true, false, trials, "matvec-bt", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, false, true, true, trials, "matvec-bt", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::fmat>(rows, true, true, false, trials, "matvec-botht", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, true, true, false, trials, "matvec-botht", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, true, true, true, trials, "matvec-botht", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::mat>(rows, false, false, false, trials, "matvec", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, false, false, false, trials, "matvec", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, false, false, true, trials, "matvec", device_name, "cuda", "double", out_file);

  run_benchmarks<arma::mat>(rows, true, false, false, trials, "matvec-at", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, true, false, false, trials, "matvec-at", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, true, false, true, trials, "matvec-at", device_name, "cuda", "double", out_file);

  run_benchmarks<arma::mat>(rows, false, true, false, trials, "matvec-bt", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, false, true, false, trials, "matvec-bt", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, false, true, true, trials, "matvec-bt", device_name, "cuda", "double", out_file);

  run_benchmarks<arma::mat>(rows, true, true, false, trials, "matvec-botht", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, true, true, false, trials, "matvec-botht", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, true, true, true, trials, "matvec-botht", device_name, "cuda", "double", out_file);
  }
