// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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



// Compute eigenvalues only into pre-given object
template<typename T1>
inline
typename
enable_if2<
    coot_real_only<typename T1::elem_type>::value,
    bool
>::result
eig_sym
  (
         Col<typename T1::elem_type>&     eigval,
  const Base<typename T1::elem_type, T1>& X
  )
  {
  coot_extra_debug_sigprint();

  // X will be destroyed during computation of the eigenvalues, and so we need a new object.
  typedef typename T1::elem_type eT;
  Mat<eT> tmp(X.get_ref());

  // check size
  if (tmp.n_rows != tmp.n_cols)
    {
    coot_debug_warm("eig_sym(): matrix must be square");
    return false;
    }

  eigval.set_size(tmp.n_rows);

  // Shortcuts for trivial cases.
  if (tmp.n_rows == 0)
    {
    return true;
    }
  else if (tmp.n_rows == 1)
    {
    eigval = tmp;
    return true;
    }

  std::tuple<bool, std::string> result = coot_rt_t::eig_sym(tmp.get_dev_mem(true), tmp.n_rows, false, eigval.get_dev_mem(true));

  if (!std::get<0>(result))
    {
    coot_debug_warn("eig_sym(): " + std::get<1>(result));
    }

  return std::get<0>(result);
  }



// Compute eigenvalues only into new object
template<typename T1>
inline
typename
enable_if2<
    coot_real_only<typename T1::elem_type>::value,
    Col<typename T1::elem_type>
>::result
eig_sym
  (
  const Base<typename T1::elem_type, T1>& X
  )
  {
  coot_extra_debug_sigprint();

  // X will be destroyed during computation of the eigenvalues, and so we need a new object.
  typedef typename T1::elem_type eT;
  Mat<eT> tmp(X.get_ref());

  // check size
  coot_debug_check( tmp.n_rows != tmp.n_cols, "eig_sym(): matrix must be square" );

  Col<eT> eigval(tmp.n_rows);

  // Shortcuts for trivial cases.
  if (tmp.n_rows == 0)
    {
    return eigval;
    }
  if (tmp.n_rows == 1)
    {
    eigval = tmp;
    return eigval;
    }

  std::tuple<bool, std::string> result = coot_rt_t::eig_sym(tmp.get_dev_mem(true), tmp.n_rows, false, eigval.get_dev_mem(true));

  if (!std::get<0>(result))
    {
    coot_stop_runtime_error("eig_sym(): " + std::get<1>(result));
    }

  return eigval;
  }



// Compute eigenvectors and eigenvalues
template<typename T1>
inline
typename
enable_if2<
    coot_real_only<typename T1::elem_type>::value,
    bool
>::result
eig_sym
  (
         Col<typename T1::elem_type>&     eigval,
         Mat<typename T1::elem_type>&     eigvec,
  const Base<typename T1::elem_type, T1>& expr,
  const char* method = "std" // for compatibility
  )
  {
  coot_extra_debug_sigprint();

  // The eigenvectors will end up in the matrix we work on.
  typedef typename T1::elem_type eT;
  eigvec = expr.get_ref();

  // check size
  if (eigvec.n_rows != eigvec.n_cols)
    {
    coot_debug_warm("eig_sym(): matrix must be square");
    return false;
    }

  eigval.set_size(eigvec.n_rows);

  // Shortcuts for trivial cases.
  if (eigvec.n_rows == 0)
    {
    return true;
    }
  else if (eigvec.n_rows == 1)
    {
    eigval = eigvec;
    eigvec[0] = (eT) 1;
    return true;
    }

  std::tuple<bool, std::string> result = coot_rt_t::eig_sym(eigvec.get_dev_mem(true), eigvec.n_rows, true, eigval.get_dev_mem(true));

  if (!std::get<0>(result))
    {
    coot_debug_warn("eig_sym(): " + std::get<1>(result));
    }

  return std::get<0>(result);
  }
