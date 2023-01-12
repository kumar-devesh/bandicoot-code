// Copyright 2021 Ryan Curtin (https://www.ratml.org_
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



// Reduce the last nb rows and columns of the symmetric/Hermitian upper
// triangular matrix A to real tridiagonal form by an orthogonal similarity
// transformation Q' * A * Q.  The matrices V and W that are needed to apply
// the transformation to the unreduced part of A are stored and returned.
template<typename eT>
void
latrd(Mat<eT>& A,
      const uword nb,
      Col<eT>& E,
      Col<eT>& tau,
      Mat<eT>& W)
  {
  coot_extra_debug_sigprint();

  // MAGMA code conversions:
  //
  //  n => A.n_rows
  //

  // Reduce last `nb` columns of the upper triangle `A`.
  for (uword i = A.n_rows - 1; i >= A.n_rows - nb; --i)
    {
    const uword i_1 = i + 1;
    const uword i_n = A.n_rows - i - 1;

    const uword iw = i - A.n_rows + nb;

    if (i < A.n_rows - 1)
      {
      // Update A(1:i, i).
      if (std::is_complex<eT>::value)
        {
        // TODO: call lacgv() here.
        }

      // dgemv:
      //   - M = i_1
      //   - N = i_n
      //   - alpha = -1
      //   - A = A(0, i+1)
      //   - lda = lda
      //   - X = W(i, iw+1)
      //   - incx = ldw
      //   - beta = 1.0
      //   - Y = A(0, i)
      //   - incy = 1
      //
      // (MxN => i_1 x i_N => (i + 1) x (A.n_rows - i - 1))
      A.col(i) = -A.col(i + 1) * W

      if (std::is_complex<eT>::value)
        {
        // TODO: two calls to lacgv() here
        }

      // dgemv: same as above but
      //    - A = W(0, iw + 1)
      //    - X = A(i, i + 1)
      A.col(i) = -W.col(iw + 1) * A(i, i + 1)??

      if (std::is_complex<eT>::value)
        {
        // TODO: another call to lacgv() here
        }
      }

    if (i > 0)
      {
      // Generate elementary reflector H(i) to annihilate A(1:i - 2, i).

      // TODO larfg()

      // Compute W(1:i - 1, i).

      // TODO symv()

      if (i < A.n_rows - 1)
        {
        // TODO gemv()

        // TODO gemv()

        // TODO gemv() with transpose

        // TODO gemv()
        }

      // TODO scal()

      // TODO dot()

      // TODO axpy()
      }
    }
  }
