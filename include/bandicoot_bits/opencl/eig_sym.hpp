// Copyright 2019 Ryan Curtin (http://www.ratml.org)
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



/**
 * Compute the eigendecomposition using OpenCL.
 */
template<typename eT>
inline
std::tuple<bool, std::string>
eig_sym(dev_mem_t<eT> mem, const uword n_rows, const bool eigenvectors, dev_mem_t<eT> eigenvalues)
  {
  coot_extra_debug_sigprint();

  if (get_rt().cl_rt.is_valid() == false)
    {
    return std::make_tuple(false, "OpenCL runtime not valid");
    }

  magma_int_t info   = 0;
  magma_int_t status = 0;
  magma_int_t lwork;
  magma_int_t liwork;
  cl_mem work_mem;
  int* iwork_mem;

  magma_vec_t jobz = (eigenvectors) ? MagmaVec : MagmaNoVec;

  // First, compute the workspace size.

  magma_int_t aux_iwork;
  if(is_float<eT>::value)
    {
    // Workspace size query.
    float aux_work;
    status = magma_ssyevd_gpu(jobz, MagmaUpper, n_rows, NULL, 0, n_rows, NULL, NULL, n_rows, &aux_work, -1, &aux_iwork, -1, &info);
    }
  else if (is_double<eT>::value)
    {
    double aux_work;
    status = magma_dsyevd_gpu(jobz, MagmaUpper, n_rows, NULL, 0, n_rows, NULL, NULL, n_rows, &aux_work, -1, &aux_iwork, -1, &info);
    }
  else
    {
    return std::make_tuple(false, "not implemented for given type: must be float or double");
    }

  if (status != MAGMA_SUCCESS)
    {
    if (info < 0)
      {
      std::ostringstream oss;
      oss << "parameter " << (-info) << " was incorrect on entry to MAGMA syevd_gpu() workspace size query";
      return std::make_tuple(false, oss.str());
      }
    else
      {
      return std::make_tuple(false, "MAGMA failure in syevd_gpu() workspace size query: " + magma::error_as_string(status));
      }
    }

  // Get workspace sizes and allocate.
  lwork = (magma_int_t) aux_work[0];
  liwork = aux_iwork[0];

  work_mem = get_rt().cl_rt.acquire_memory<eT>(iwork);
  iwork_mem = cpu_memory::acquire<int>(liwork);

  if (is_float<eT>::value)
    {
    status = magma_ssyevd_gpu(jobz, MagmaUpper, n_rows, mem.cl_mem_ptr, n_rows, eigenvalues.cl_mem_ptr, work_mem, lwork, iwork_mem, liwork, &info);
    }
  else if(is_double<eT>::value)
    {
    status = magma_dsyevd_gpu(jobz, MagmaUpper, n_rows, mem.cl_mem_ptr, n_rows, eigenvalues.cl_mem_ptr, work_mem, lwork, iwork_mem, liwork, &info);
    }
  else
    {
    return std::make_tuple(false, "not implemented for given type; must be float or double");
    }

  // Process the returned info.
  if (status != MAGMA_SUCCESS)
    {
    if (info < 0)
      {
      std::ostringstream oss;
      oss << "parameter " << (-info) << " was incorrect on entry to MAGMA syevd_gpu()";
      return std::make_tuple(false, oss.str());
      }
    else if (info > 0)
      {
      std::ostringstream oss;
      if (eigenvectors)
        {
        oss << "eigendecomposition failed: could not compute an eigenvalue while working on block submatrix " << info;
        }
      else
        {
        oss << "eigendecomposition failed: " << info << " off-diagonal elements of an intermediate tridiagonal form did not converge to 0";
        }
      return std::make_tuple(false, oss.str());
      }
    }

  return std::make_tuple(true, "");
  }
