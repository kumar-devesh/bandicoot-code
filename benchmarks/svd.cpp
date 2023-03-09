/**
 * Benchmark comparison of Cholesky decomposition between CUDA and OpenCL backends.
 */
#include <bandicoot>
#include <armadillo>

using namespace coot;



template<typename MatType, typename ColType>
double run_benchmark(const uword rows, const uword cols, const bool cuda, const bool compute_u_v)
  {
  // set the correct backend
  if (cuda)
    get_rt().backend = CUDA_BACKEND;
  else
    get_rt().backend = CL_BACKEND;

  MatType x;
  x.randu(rows, cols);

  ColType s;
  MatType u, v;

  // finish all delayed operations
  get_rt().synchronise();

  double t;
  if (compute_u_v)
    {
    wall_clock c;

    c.tic();
    const bool status = svd(u, s, v, x);
    get_rt().synchronise();
    t = c.toc();

    if (!status)
      std::cerr << "svd() failed!\n";

    // sanity check
    MatType ds(u.n_cols, v.n_rows);
    ds.zeros();
    for (uword i = 0; i < s.n_elem; ++i)
      ds(i, i) = typename MatType::elem_type(s[i]);

    MatType y = (u * ds * v.t());
    y -= x;
    const typename MatType::elem_type diff = accu(square(y)) / y.n_elem;
    if (diff > 1e-5)
      std::cerr << "reconstruction diff too high: " << diff << "\n";
    }
  else
    {
    wall_clock c;

    c.tic();
    const bool status = svd(s, x);
    get_rt().synchronise();
    t = c.toc();

    if (!status)
      std::cerr << "svd() failed!\n";
    }

  return t;
  }



template<typename MatType, typename ColType>
void run_benchmarks(const uword rows,
                    const uword cols,
                    const bool compute_uv,
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
    const double t = run_benchmark<MatType, ColType>(rows, cols, cuda, compute_uv);

    out << task_name << "," << device_name << "," << backend_name << "," << elem_type << ","
        << rows << "," << rows << "," << trial << "," << t << "\n";
    std::cout << task_name << ", " << device_name << ", " << backend_name << ", " << elem_type << ", "
        << rows << ", " << rows << ", " << trial << ", " << t << "\n";
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

  std::cout << "svd: singular value decomposition benchmark comparison\n";
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

  run_benchmarks<arma::fmat, arma::fvec>(rows, cols, false, false, trials, "svd_no_uv", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat, coot::fvec>(rows, cols, false, false, trials, "svd_no_uv", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat, coot::fvec>(rows, cols, false, true, trials, "svd_no_uv", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::mat, arma::vec>(rows, cols, false, false, trials, "svd_no_uv", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat, coot::vec>(rows, cols, false, false, trials, "svd_no_uv", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat, coot::vec>(rows, cols, false, true, trials, "svd_no_uv", device_name, "cuda", "double", out_file);

  run_benchmarks<arma::fmat, arma::fvec>(rows, cols, true, false, trials, "svd_full", device_name, "cpu", "float", out_file);
  run_benchmarks<coot::fmat, coot::fvec>(rows, cols, true, false, trials, "svd_full", device_name, "opencl", "float", out_file);
  run_benchmarks<coot::fmat, coot::fvec>(rows, cols, true, true, trials, "svd_full", device_name, "cuda", "float", out_file);

  run_benchmarks<arma::mat, arma::vec>(rows, cols, true, false, trials, "svd_full", device_name, "cpu", "double", out_file);
  run_benchmarks<coot::mat, coot::vec>(rows, cols, true, false, trials, "svd_full", device_name, "opencl", "double", out_file);
  run_benchmarks<coot::mat, coot::vec>(rows, cols, true, true, trials, "svd_full", device_name, "cuda", "double", out_file);
  }
