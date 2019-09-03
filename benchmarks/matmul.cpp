/**
 * Benchmark comparison of matrix multiplication between CUDA and OpenCL backends.
 */
#include <bandicoot>

using namespace coot;

template<typename eT>
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

  Mat<eT> x(size, size);
  Mat<eT> y(size, size);

  x.fill(eT(1));
  y.fill(eT(2));

  Mat<eT> z(size, size);

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

template<typename eT>
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

  Mat<eT> x(rank, size);
  Mat<eT> y(size, rank);

  x.fill(eT(1));
  y.fill(eT(2));

  Mat<eT> z(rank, rank);

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
  const double pow_min = 2.0;
  const double pow_step = 0.25;

  for (double d = pow_min; d <= 4.0; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = float, square matrices, dim = " << size << ": ";

    std::cout << "OpenCL: " << benchmark_square<float>(size, false, false, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<float>(size, false, false, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = double, square matrices, dim = " << size << ": ";

    std::cout << "OpenCL: " << benchmark_square<double>(size, false, false, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<double>(size, false, false, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";
/*
  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y.t(), eT = float, square matrices, dim = " << size << ": ";

    std::cout << "OpenCL: " << benchmark_square<float>(size, true, false, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<float>(size, true, false, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y.t(), eT = double, square matrices, dim = " << size << ": ";
    std::cout << "OpenCL: " << benchmark_square<double>(size, false, true, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<double>(size, false, true, CUDA_BACKEND) << "s\n";

    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x.t() * y, eT = float, square matrices, dim = " << size << ": ";

    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x.t() * y, eT = double, square matrices, dim = " << size << ": ";
    std::cout << "OpenCL: " << benchmark_square<float>(size, true, true, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<float>(size, true, true, CUDA_BACKEND) << "s\n";

    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x.t() * y.t(), eT = float, square matrices, dim = " << size << ": ";

    std::cout << "OpenCL: " << benchmark_square<double>(size, true, true, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_square<double>(size, true, true, CUDA_BACKEND) << "s\n";

    }

  std::cout << "\n";

  for (double d = pow_min; d <= pow_max; d += pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x.t() * y, eT = double, square matrices, dim = " << size << ": ";

    }*/

  for (double d = pow_min; d <= 1.5 * pow_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = float, dim = [" << size << " x 10]: ";

    std::cout << "OpenCL: " << benchmark_low_rank<float>(size, 10, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<float>(size, 10, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= 1.5 * pow_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = double, dim = [" << size << " x 10]: ";

    std::cout << "OpenCL: " << benchmark_low_rank<double>(size, 10, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<double>(size, 10, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= 1.5 * pow_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = float, dim = [" << size << " x 50]: ";

    std::cout << "OpenCL: " << benchmark_low_rank<float>(size, 50, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<float>(size, 50, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= 1.5 * pow_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = double, dim = [" << size << " x 50]: ";

    std::cout << "OpenCL: " << benchmark_low_rank<double>(size, 50, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<double>(size, 50, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= 1.5 * pow_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = float, dim = [" << size << " x 100]: ";

    std::cout << "OpenCL: " << benchmark_low_rank<float>(size, 100, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<float>(size, 100, CUDA_BACKEND) << "s\n";
    }

  std::cout << "\n";

  for (double d = pow_min; d <= 1.5 * pow_max; d += 2.0 * pow_step)
    {
    const uword size = uword(std::pow(10.0, d));

    std::cout << "x * y, eT = double, dim = [" << size << " x 100]: ";

    std::cout << "OpenCL: " << benchmark_low_rank<double>(size, 100, CL_BACKEND) << "s, ";
    std::cout << "CUDA: " << benchmark_low_rank<double>(size, 100, CUDA_BACKEND) << "s\n";
    }
  }
