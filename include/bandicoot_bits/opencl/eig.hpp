// Copyright 2021 Ryan Curtin (http://www.ratml.org/)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


//! \addtogroup opencl
//! @{

/**
 * Compute the eigendecomposition of the given matrix, storing eigenvalues and eigenvectors.
 */
template<typename eT>
inline
void
eig_sym(dev_mem_t<eT> eigval, dev_mem_t<eT> eigvec, const dev_mem_t<eT> A, const uword n_rows)
  {
  // We assume that the output is the right size already, and that the matrix is symmetric.
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "opencl::eig_sym(): OpenCL runtime not valid");

  // This implementation is a transcription and adaptation of clMAGMA's Xsyevd().
  // (Thank you to the authors of that package!)
  //
  // Variable names typically reflect the same variable names used there.

  /**
   * Here are our input parameters:
   *
   * - jobz: `V` in our case, since we are computing eigenvectors
   * - uplo: `U` (arbitrary)
   * - N: `n_rows`
   * - A: `A`
   * - LDA: `n_rows`
   * - W: `eigval`
   * - WORK: temporary workspace; size must be computed internally
   * - LWORK: size, from magma_get_dsytrd_nb()?
   * - IWORK: temporary integer workspace, size must be computed internally
   * - LIWORK: size, computed
   * - INFO: we can ignore this
   */

  // Note that these parameter is tunable.  (Perhaps even auto-tunable.)
  // The clMAGMA sources are tuned for Tahiti-architecture cards, and I took their values directly.
  const uword nb = 32; // From magma_get_Xsytrd_nb().
  const uword lwork = max(2 * n_rows + n_rows * nb, 1 + 6 * n_rows + 2 * n_rows * n_rows);
  const uword liwork = 3 + 5 * n_rows;

  // Shortcut case: when there's only one element, we already have the solution.
  if (n_rows == 1)
    {
    copy_array(eigval, A, 1);
    inplace_op_scalar(eigvec, eT(1), 1, oneway_kernel_id::inplace_set_scalar);

    return;
    }

  // First, move A to our working memory (eigvec).
  copy_array(eigvec, A, n_rows * n_rows);

  // We don't shortcut to the CPU, since we already have the matrix stored on the GPU.
  // (This is unlike clMAGMA, which copies a matrix to the GPU when it is sufficiently large.)

  // Scale matrix to allowable range, if needed.

  // Compute the maximum-norm element.
  const eT anrm = TODO;
  // sfmin / eps
  // TODO: what's sfmin?  eps is numeric_limits<eT>::epsilon()
  const eT rmin = sqrt(sfmin / eps) // TODO
  const eT rmax = sqrt(eps / sfmin)
  const eT sigma = eT(1);
  if (anrm > eT(0) && anrm < rmin)
    sigma = rmin / anrm;
  else if (anrm > rmax)
    sigma = rmax / anrm;

  if (sigma != eT(1))
    {
    // TODO: we could have a custom kernel that only multiplied the upper triangular portion here.
    inplace_op_scalar(eigvec, sigma, n_rows * n_rows, oneway_kernel_id::inplace_mul_scalar);
    }

  // Use the Xsytrd() algorithm to reduce the symmetric matrix to a tridiagonal form.
  // Xsytrd work: e (n) + tau (n) + llwork (n*nb) ==> 2n + n*nb
  // Xstedx work: e (n) + tau (n) + z (n*N) + llwrk2 (1 + 4*n + n^2) ==> 1 + 6n + 2n^2

  /**
   * Xsytrd(...)
   */

  // For eigenvectors, first use the Xstedc() algorithm to generate the eigenvector
  // matrix, `work`, of the tridiagonal matrix, then use the Xormtr() algorithm to
  // multiply it to the Householder transformations represented as Householder vectors
  // in A.

  /**
   * Xstedx(...);
   * Xomtr(...);
   * Xlacpy(...);
   */

  // If matrix was scaled, then rescale eigenvalues appropriately.

  /**
   * Xscal(...);
   */

  return;
  }

//! @}
