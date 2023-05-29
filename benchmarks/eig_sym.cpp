/**
 * Benchmark comparison of eigendecomposition between CUDA and OpenCL backends.
 */
#include <bandicoot>
#include <armadillo>

using namespace coot;



template<typename MatType>
void fill_randu(MatType& x)
  {
  arma::Mat<typename MatType::elem_type> cpu_x(x.n_rows, x.n_cols);

  cpu_x.randu();

  // But it needs to be symmetric positive definite...
  cpu_x = cpu_x.t() * cpu_x;
  cpu_x.diag() += 3;

  x.copy_into_dev_mem(cpu_x.memptr(), x.n_elem);
  }



template<>
void fill_randu(arma::fmat& x)
  {
  x.randu();
  x = x.t() * x;
  x.diag() += 3;
  }



template<>
void fill_randu(arma::mat& x)
  {
  x.randu();
  x = x.t() * x;
  x.diag() += 3;
  }



template<typename MatType, typename ColType>
double run_benchmark(const uword rows, const bool cuda, const bool eigenvectors)
  {
  // set the correct backend
  if (cuda)
    get_rt().backend = CUDA_BACKEND;
  else
    get_rt().backend = CL_BACKEND;

  MatType x;
  x.set_size(rows, rows);
  fill_randu(x);

  ColType eigvals;
  MatType eigvecs;

  // finish all delayed operations
  get_rt().synchronise();

  wall_clock c;
  double t;

  c.tic();
  bool status;
  if (eigenvectors)
    status = eig_sym(eigvals, eigvecs, x);
  else
    status = eig_sym(eigvals, x);

  get_rt().synchronise();
  t = c.toc();

  if (!status)
    std::cerr << "eig_sym() failed!\n";

  // sanity check
  if (eigenvectors)
    {
    MatType y = (eigvecs * diagmat(eigvals) * eigvecs.t());
    y -= x;
    const typename MatType::elem_type diff = accu(square(y)) / y.n_elem;
    if (diff > 1e-3)
      std::cerr << "reconstruction diff too high: " << diff << "\n";
    }

  return t;
  }



template<typename MatType, typename ColType>
void run_benchmarks(const uword rows,
                    const bool eigenvectors,
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
    const double t = run_benchmark<MatType, ColType>(rows, cuda, eigenvectors);

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

  std::cout << "eig_sym: eigendecomposition benchmark comparison\n";
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

  run_benchmarks<arma::fmat, arma::fvec>(rows, false, false, trials, "eig_sym_1", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat, coot::fvec>(rows, false, false, trials, "eig_sym_1", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat, coot::fvec>(rows, false, true, trials, "eig_sym_1", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::mat, arma::vec>(rows, false, false, trials, "eig_sym_1", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat, coot::vec>(rows, false, false, trials, "eig_sym_1", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat, coot::vec>(rows, false, true, trials, "eig_sym_1", device_name, "cuda", "double", out_file);

  // now compute eigenvectors too
  run_benchmarks<arma::fmat, arma::fvec>(rows, true, false, trials, "eig_sym_2", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat, coot::fvec>(rows, true, false, trials, "eig_sym_2", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat, coot::fvec>(rows, true, true, trials, "eig_sym_2", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::mat, arma::vec>(rows, true, false, trials, "eig_sym_2", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat, coot::vec>(rows, true, false, trials, "eig_sym_2", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat, coot::vec>(rows, true, true, trials, "eig_sym_2", device_name, "cuda", "double", out_file);
  }
