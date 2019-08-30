// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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

// utility functions for compiled-on-the-fly CUDA kernels

// shitty single kernel for fill()

inline
std::vector<std::string>
get_cuda_kernel_names()
  {
  std::vector<std::string> names;

  names.push_back("inplace_set_scalar");
  names.push_back("inplace_plus_scalar");
  names.push_back("inplace_minus_scalar");
  names.push_back("inplace_mul_scalar");
  names.push_back("inplace_div_scalar");

  names.push_back("submat_inplace_set_scalar");
  names.push_back("submat_inplace_plus_scalar");
  names.push_back("submat_inplace_minus_scalar");
  names.push_back("submat_inplace_mul_scalar");
  names.push_back("submat_inplace_div_scalar");

  names.push_back("inplace_plus_array");
  names.push_back("inplace_minus_array");
  names.push_back("inplace_mul_array");
  names.push_back("inplace_div_array");

  names.push_back("submat_inplace_set_mat");
  names.push_back("submat_inplace_plus_mat");
  names.push_back("submat_inplace_minus_mat");
  names.push_back("submat_inplace_schur_mat");
  names.push_back("submat_inplace_div_mat");

  names.push_back("equ_array_plus_scalar");
  names.push_back("equ_array_neg");
  names.push_back("equ_array_minus_scalar_pre");
  names.push_back("equ_array_minus_scalar_post");
  names.push_back("equ_array_mul_scalar");
  names.push_back("equ_array_div_scalar_pre");
  names.push_back("equ_array_div_scalar_post");
  names.push_back("equ_array_square");
  names.push_back("equ_array_sqrt");
  names.push_back("equ_array_exp");
  names.push_back("equ_array_log");

  names.push_back("equ_array_plus_array");
  names.push_back("equ_array_minus_array");
  names.push_back("equ_array_mul_array");
  names.push_back("equ_array_div_array");

  return names;
  }

inline
std::string
get_cuda_kernel_src()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::string source = \

  "#define uint unsigned int\n"
  "#define COOT_FN2(ARG1, ARG2)  ARG1 ## ARG2 \n"
  "#define COOT_FN(ARG1,ARG2) COOT_FN2(ARG1,ARG2)\n"
  "\n"
  "extern \"C\" {\n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_set_scalar)(eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_plus_scalar)(eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] += val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_minus_scalar)(eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] -= val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_mul_scalar)(eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] *= val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_div_scalar)(eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] /= val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,submat_inplace_set_scalar)(eT* out, const eT val, const UWORD end_row, const UWORD end_col, const UWORD n_rows) \n"
  "  { \n"
  "  const UWORD row = blockIdx.x; \n" // TODO: check this isn't backwards
  "  const UWORD col = threadIdx.x; \n"
  "  if ((row <= end_row) && (col <= end_col)) \n"
  "    { out[row + col * n_rows] = val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,submat_inplace_plus_scalar)(eT* out, const eT val, const UWORD end_row, const UWORD end_col, const UWORD n_rows) \n"
  "  { \n"
  "  const UWORD row = blockIdx.x; \n" // TODO: check this isn't backwards
  "  const UWORD col = threadIdx.x; \n"
  "  if ((row <= end_row) && (col <= end_col)) \n"
  "    { out[row + col * n_rows] += val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,submat_inplace_minus_scalar)(eT* out, const eT val, const UWORD end_row, const UWORD end_col, const UWORD n_rows) \n"
  "  { \n"
  "  const UWORD row = blockIdx.x; \n" // TODO: check this isn't backwards
  "  const UWORD col = threadIdx.x; \n"
  "  if ((row <= end_row) && (col <= end_col)) \n"
  "    { out[row + col * n_rows] -= val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,submat_inplace_mul_scalar)(eT* out, const eT val, const UWORD end_row, const UWORD end_col, const UWORD n_rows) \n"
  "  { \n"
  "  const UWORD row = blockIdx.x; \n" // TODO: check this isn't backwards
  "  const UWORD col = threadIdx.x; \n"
  "  if ((row <= end_row) && (col <= end_col)) \n"
  "    { out[row + col * n_rows] *= val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,submat_inplace_div_scalar)(eT* out, const eT val, const UWORD end_row, const UWORD end_col, const UWORD n_rows) \n"
  "  { \n"
  "  const UWORD row = blockIdx.x; \n" // TODO: check this isn't backwards
  "  const UWORD col = threadIdx.x; \n"
  "  if ((row <= end_row) && (col <= end_col)) \n"
  "    { out[row + col * n_rows] /= val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_plus_array)(eT* out, const eT* A, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if (i < N) { out[i] += A[i]; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_minus_array)(eT* out, const eT* A, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if (i < N) { out[i] -= A[i]; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_mul_array)(eT* out, const eT* A, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if (i < N) { out[i] *= A[i]; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_div_array)(eT* out, const eT* A, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if (i < N) { out[i] /= A[i]; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,submat_inplace_set_mat)(eT* out, const eT* A, const UWORD out_start_row, const UWORD out_start_col, const UWORD out_n_rows, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = blockIdx.x; \n"  // row in source matrix
  "  const UWORD col = threadIdx.x; \n"  // col in source matrix
  "  if( (row <= A_n_rows) && (col <= A_n_cols) ) \n"
  "    { \n"
  "    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows); \n"
  "    const UWORD   A_index = row + col*A_n_rows; \n"
  "    out[out_index] = A[A_index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,submat_inplace_plus_mat)(eT* out, const eT* A, const UWORD out_start_row, const UWORD out_start_col, const UWORD out_n_rows, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = blockIdx.x; \n"  // row in source matrix
  "  const UWORD col = threadIdx.x; \n"  // col in source matrix
  "  if( (row <= A_n_rows) && (col <= A_n_cols) ) \n"
  "    { \n"
  "    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows); \n"
  "    const UWORD   A_index = row + col*A_n_rows; \n"
  "    out[out_index] += A[A_index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,submat_inplace_minus_mat)(eT* out, const eT* A, const UWORD out_start_row, const UWORD out_start_col, const UWORD out_n_rows, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = blockIdx.x; \n"  // row in source matrix
  "  const UWORD col = threadIdx.x; \n"  // col in source matrix
  "  if( (row <= A_n_rows) && (col <= A_n_cols) ) \n"
  "    { \n"
  "    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows); \n"
  "    const UWORD   A_index = row + col*A_n_rows; \n"
  "    out[out_index] -= A[A_index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,submat_inplace_schur_mat)(eT* out, const eT* A, const UWORD out_start_row, const UWORD out_start_col, const UWORD out_n_rows, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = blockIdx.x; \n"  // row in source matrix
  "  const UWORD col = threadIdx.x; \n"  // col in source matrix
  "  if( (row <= A_n_rows) && (col <= A_n_cols) ) \n"
  "    { \n"
  "    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows); \n"
  "    const UWORD   A_index = row + col*A_n_rows; \n"
  "    out[out_index] *= A[A_index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,submat_inplace_div_mat)(eT* out, const eT* A, const UWORD out_start_row, const UWORD out_start_col, const UWORD out_n_rows, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = blockIdx.x; \n"  // row in source matrix
  "  const UWORD col = threadIdx.x; \n"  // col in source matrix
  "  if( (row <= A_n_rows) && (col <= A_n_cols) ) \n"
  "    { \n"
  "    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows); \n"
  "    const UWORD   A_index = row + col*A_n_rows; \n"
  "    out[out_index] /= A[A_index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_plus_scalar)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = A[i] + val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_neg)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  (void)(val); \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = -(A[i]); } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_minus_scalar_pre)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = val - A[i]; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_minus_scalar_post)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = A[i] - val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_mul_scalar)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = A[i] * val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_div_scalar_pre)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = val / A[i]; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_div_scalar_post)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = A[i] / val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_square)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  (void)(val); \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { const eT Ai = A[i]; out[i] = Ai * Ai; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_sqrt)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  (void)(val); \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = (eT)sqrt( (promoted_eT)(A[i]) ); } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_exp)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  (void)(val); \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = (eT)exp( (promoted_eT)(A[i]) ); } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_log)(eT* out, const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  (void)(val); \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = (eT)log( (promoted_eT)(A[i]) ); } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_plus_array)(eT* out, const eT* A, const eT* B, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = A[i] + B[i]; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_minus_array)(eT* out, const eT* A, const eT* B, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = A[i] - B[i]; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_mul_array)(eT* out, const eT* A, const eT* B, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = A[i] * B[i]; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,equ_array_div_array)(eT* out, const eT* A, const eT* B, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = A[i] / B[i]; } \n"
  "  } \n"
  "\n"
  "}\n";

  return source;
  }
