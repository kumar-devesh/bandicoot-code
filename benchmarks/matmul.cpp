/**
 * Benchmark comparison of matrix multiplication between CUDA and OpenCL backends.
 */
#include <bandicoot>
#include <armadillo>

using namespace coot;

template<typename MatType>
double benchmark_square(const uword size, const bool trans_a, const bool trans_b, const bool opencl)
  {
  if (opencl)
    {
    coot::get_rt().backend = CL_BACKEND;
    }
  else
    {
    coot::get_rt().backend = CUDA_BACKEND;
    }

  MatType x(size, size);
  MatType y(size, size);

  x.fill(typename MatType::elem_type(1));
  y.fill(typename MatType::elem_type(2));

  MatType z(size, size);

  // Try to catch outliers: return the min of 3 trials.
  // TODO: seems like that always gives garbage for CUDA... is it caching results or something?

  wall_clock c;
  double best_t = std::numeric_limits<double>::max();
  for (size_t trial = 0; trial < 1; ++trial)
    {
    double t = 0.0;
    if (trans_a && trans_b)
      {
      c.tic();
      z = x.t() * y.t();
      t = c.toc();
      }
    else if (trans_a)
      {
      c.tic();
      z = x.t() * y;
      t = c.toc();
      }
    else if (trans_b)
      {
      c.tic();
      z = x * y.t();
      t = c.toc();
      }
    else
      {
      c.tic();
      z = x * y;
      t = c.toc();
      }

    if (t < best_t)
      {
      best_t = t;
      }
    }

  return best_t;
  }

template<typename MatType>
double benchmark_low_rank(const uword size, const uword rank, const bool opencl)
  {
  if (opencl)
    {
    coot::get_rt().backend = CL_BACKEND;
    }
  else
    {
    coot::get_rt().backend = CUDA_BACKEND;
    }

  MatType x(rank, size);
  MatType y(size, rank);

  x.fill(typename MatType::elem_type(1));
  y.fill(typename MatType::elem_type(2));

  MatType z(rank, rank);

  wall_clock c;
  double best_t = std::numeric_limits<double>::max();
  for (size_t trial = 0; trial < 1; ++trial)
    {
    double t = 0.0;

    c.tic();
    z = x * y;
    t = c.toc();

    if (t < best_t)
      {
      best_t = t;
      }
    }

  return best_t;
  }

int main()
  {
  wall_clock c;

  std::cout << "matmul: Matrix multiplication benchmark comparison\n";
  std::cout << "  bandicoot version " << coot::coot_version::as_string() << '\n';
  std::cout << "  armadillo version " << arma::arma_version::as_string() << '\n';
  std::cout << '\n';

  // Time initialization of each.
  c.tic();
  coot::get_rt().cl_rt.init(true);
  double time = c.toc();
  std::cout << "OpenCL initialization time: " << time << "s\n";

  c.tic();
  coot::get_rt().cuda_rt.init();
  time = c.toc();

  coot::get_rt().backend = CUDA_BACKEND;

  std::cout << "CUDA initialization time: " << time << "s\n";
  std::cout << "\n";

  const double pow_max = 4.0;
  const double pow_lr_max = 6.5;
  const double pow_min = 2.0;
  const double pow_step = 0.25;

  for (double d = pow_min; d <= 4.0; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = float, square matrices, dim = " << size << ": ";

    std::cout << "arma: " << benchmark_square<arma::fmat>(size, false, false, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_square<fmat>(size, false, false, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<fmat>(size, false, false, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = double, square matrices, dim = " << size << ": ";

    std::cout << "arma: " << benchmark_square<arma::mat>(size, false, false, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_square<mat>(size, false, false, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<mat>(size, false, false, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";
/*
  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y.t(), eT = float, square matrices, dim = " << size << ": ";

    std::cout << "arma: " << benchmark_square<arma::fmat>(size, false, true, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_square<fmat>(size, false, true, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<fmat>(size, false, true, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y.t(), eT = double, square matrices, dim = " << size << ": ";

    std::cout << "arma: " << benchmark_square<arma::mat>(size, false, true, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_square<mat>(size, false, true, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<mat>(size, false, true, CUDA_BACKEND) << "s\n";

    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x.t() * y, eT = float, square matrices, dim = " << size << ": ";

    std::cout << "arma: " << benchmark_square<arma::fmat>(size, true, false, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_square<fmat>(size, true, false, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<fmat>(size, true, false, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x.t() * y, eT = double, square matrices, dim = " << size << ": ";

    std::cout << "arma: " << benchmark_square<arma::mat>(size, true, false, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_square<mat>(size, true, false, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<mat>(size, true, false, CUDA_BACKEND) << "s\n";

    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x.t() * y.t(), eT = float, square matrices, dim = " << size << ": ";

    std::cout << "arma: " << benchmark_square<arma::fmat>(size, true, true, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_square<fmat>(size, true, true, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<fmat>(size, true, true, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x.t() * y, eT = double, square matrices, dim = " << size << ": ";

    std::cout << "arma: " << benchmark_square<arma::mat>(size, true, true, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_square<mat>(size, true, true, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<mat>(size, true, true, CUDA_BACKEND) << "s\n";
    }*/

  for (double d = pow_min; d <= pow_lr_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = float, dim = [" << size << " x 10]: ";

    std::cout << "arma: " << benchmark_low_rank<arma::fmat>(size, 10, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_low_rank<fmat>(size, 10, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<fmat>(size, 10, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_lr_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = double, dim = [" << size << " x 10]: ";

    std::cout << "arma: " << benchmark_low_rank<arma::mat>(size, 10, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_low_rank<mat>(size, 10, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<mat>(size, 10, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_lr_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = float, dim = [" << size << " x 50]: ";

    std::cout << "arma: " << benchmark_low_rank<arma::fmat>(size, 50, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_low_rank<fmat>(size, 50, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<fmat>(size, 50, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_lr_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = double, dim = [" << size << " x 50]: ";

    std::cout << "arma: " << benchmark_low_rank<arma::mat>(size, 50, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_low_rank<mat>(size, 50, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<mat>(size, 50, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_lr_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = float, dim = [" << size << " x 100]: ";

    std::cout << "arma: " << benchmark_low_rank<arma::fmat>(size, 100, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_low_rank<fmat>(size, 100, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<fmat>(size, 100, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_lr_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = double, dim = [" << size << " x 100]: ";

    std::cout << "arma: " << benchmark_low_rank<arma::mat>(size, 100, CL_BACKEND) << "s, ";
    std::cout << "OpenCL: " << benchmark_low_rank<mat>(size, 100, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<mat>(size, 100, CUDA_BACKEND) << "s\n";
    }
  }
