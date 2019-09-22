/**
 * Benchmark comparison of matrix multiplication between CUDA and OpenCL backends.
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
void check_matmul(MatType& x, MatType& y, MatType& z, const bool trans_a, const bool trans_b)
  {
  arma::Mat<typename MatType::elem_type> cpu_x, cpu_y, cpu_zz, cpu_z;
  cpu_x.set_size(x.n_rows, x.n_cols);
  cpu_y.set_size(y.n_rows, y.n_cols);
  cpu_z.set_size(z.n_rows, z.n_cols);

  x.copy_from_dev_mem(cpu_x.memptr(), x.n_elem);
  y.copy_from_dev_mem(cpu_y.memptr(), y.n_elem);
  z.copy_from_dev_mem(cpu_z.memptr(), z.n_elem);

  if (!trans_a && !trans_b)
    cpu_zz = cpu_x * cpu_y;
  else if (trans_a && !trans_b)
    cpu_zz = cpu_x.t() * cpu_y;
  else if (!trans_a && trans_b)
    cpu_zz = cpu_x * cpu_y.t();
  else
    cpu_zz = cpu_x.t() * cpu_y.t();

  const double diff = arma::norm(cpu_z - cpu_zz, 2) / cpu_z.n_elem;
  if (diff > 1e-7)
    {
    std::cerr << "failed diff check! " << diff << "\n";
    }
  }



template<>
void check_matmul(arma::fmat& x, arma::fmat& y, arma::fmat& z, const bool trans_a, const bool trans_b)
  {
  // nothing to do
  }



template<>
void check_matmul(arma::mat& x, arma::mat& y, arma::mat& z, const bool trans_a, const bool trans_b)
  {
  // nothing to do
  }



template<typename MatType>
double run_benchmark(const uword rows, const uword cols, const bool trans_a, const bool trans_b, const bool cuda)
  {
  // set the correct backend
  if (cuda)
    get_rt().backend = CUDA_BACKEND;
  else
    get_rt().backend = CL_BACKEND;

  MatType x;
  if (trans_a)
    x.set_size(cols, rows);
  else
    x.set_size(rows, cols);

  MatType y;
  if (trans_b)
    y.set_size(rows, cols);
  else
    y.set_size(cols, rows);

  fill_randu(x, y);

  MatType z(rows, rows);

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

  // sanity check on backend
  //if (!std::is_same<MatType, arma::fmat>::value && !std::is_same<MatType, arma::mat>::value)
  //  {
  //  check_matmul<MatType>(x, y, z, trans_a, trans_b);
  //  }

  return t;
  }



template<typename MatType>
void run_benchmarks(const uword rows,
                    const uword cols,
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
    const double t = run_benchmark<MatType>(rows, cols, trans_a, trans_b, cuda_backend);

    out << task_name << ", " << device_name << ", " << backend_name << ", " << elem_type << ", "
        << rows << ", " << cols << ", " << trial << ", " << t << "\n";
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

  std::cout << "matmul: Matrix multiplication benchmark comparison\n";
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
  run_benchmarks<arma::fmat>(rows, cols, false, false, false, trials, "matmul", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, cols, false, false, false, trials, "matmul", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, cols, false, false, true, trials, "matmul", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::fmat>(rows, cols, true, false, false, trials, "matmul-at", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, cols, true, false, false, trials, "matmul-at", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, cols, true, false, true, trials, "matmul-at", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::fmat>(rows, cols, false, true, false, trials, "matmul-bt", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, cols, false, true, false, trials, "matmul-bt", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, cols, false, true, true, trials, "matmul-bt", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::fmat>(rows, cols, true, true, false, trials, "matmul-botht", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat>(rows, cols, true, true, false, trials, "matmul-botht", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat>(rows, cols, true, true, true, trials, "matmul-botht", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::mat>(rows, cols, false, false, false, trials, "matmul", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, cols, false, false, false, trials, "matmul", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, cols, false, false, true, trials, "matmul", device_name, "cuda", "double", out_file);

  run_benchmarks<arma::mat>(rows, cols, true, false, false, trials, "matmul-at", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, cols, true, false, false, trials, "matmul-at", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, cols, true, false, true, trials, "matmul-at", device_name, "cuda", "double", out_file);

  run_benchmarks<arma::mat>(rows, cols, false, true, false, trials, "matmul-bt", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, cols, false, true, false, trials, "matmul-bt", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, cols, false, true, true, trials, "matmul-bt", device_name, "cuda", "double", out_file);

  run_benchmarks<arma::mat>(rows, cols, true, true, false, trials, "matmul-botht", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat>(rows, cols, true, true, false, trials, "matmul-botht", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat>(rows, cols, true, true, true, trials, "matmul-botht", device_name, "cuda", "double", out_file);
  }
