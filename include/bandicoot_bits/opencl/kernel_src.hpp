// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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



struct kernel_src
  {
  static inline const std::string&  get_source();
  static inline       std::string  init_source();
  };



inline
const std::string&
kernel_src::get_source()
  {
  static const std::string source = init_source();
  
  return source;
  }



// TODO: inplace_set_scalar() could be replaced with explicit call to clEnqueueFillBuffer()
// present in OpenCL 1.2: http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueFillBuffer.html

// TODO: need submat analogues of all functions

// TODO: need specialised handling for cx_float and cx_double
// for example (cx_double * cx_double) is not simply (double2 * double2)
  

inline
std::string
kernel_src::init_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct
  
  std::string source = \
  "#ifdef cl_khr_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_khr_pragma_unroll : enable \n"
  "#endif \n"
  "#ifdef cl_amd_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_amd_pragma_unroll : enable \n"
  "#endif \n"
  "#ifdef cl_nv_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable \n"
  "#endif \n"
  "#ifdef cl_intel_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_intel_pragma_unroll : enable \n"
  "#endif \n"
  "\n"
  "#define COOT_FN2(ARG1,ARG2)  ARG1 ## ARG2 \n"
  "#define COOT_FN(ARG1,ARG2) COOT_FN2(ARG1,ARG2) \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,inplace_set_scalar)(__global eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,inplace_plus_scalar)(__global eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] += val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,inplace_minus_scalar)(__global eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] -= val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,inplace_mul_scalar)(__global eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] *= val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,inplace_div_scalar)(__global eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] /= val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_inplace_set_scalar)(__global eT* out, const eT val, const UWORD end_row, const UWORD end_col, const UWORD n_rows) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"  // row in the parent matrix
  "  const UWORD col = get_global_id(1); \n"  // col in the parent matrix
  "  if( (row <= end_row) && (col <= end_col) ) \n"
  "    { out[row + col*n_rows] = val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_inplace_plus_scalar)(__global eT* out, const eT val, const UWORD end_row, const UWORD end_col, const UWORD n_rows) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"  // row in the parent matrix
  "  const UWORD col = get_global_id(1); \n"  // col in the parent matrix
  "  if( (row <= end_row) && (col <= end_col) ) \n"
  "    { out[row + col*n_rows] += val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_inplace_minus_scalar)(__global eT* out, const eT val, const UWORD end_row, const UWORD end_col, const UWORD n_rows) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"  // row in the parent matrix
  "  const UWORD col = get_global_id(1); \n"  // col in the parent matrix
  "  if( (row <= end_row) && (col <= end_col) ) \n"
  "    { out[row + col*n_rows] -= val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_inplace_mul_scalar)(__global eT* out, const eT val, const UWORD end_row, const UWORD end_col, const UWORD n_rows) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"  // row in the parent matrix
  "  const UWORD col = get_global_id(1); \n"  // col in the parent matrix
  "  if( (row <= end_row) && (col <= end_col) ) \n"
  "    { out[row + col*n_rows] *= val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_inplace_div_scalar)(__global eT* out, const eT val, const UWORD end_row, const UWORD end_col, const UWORD n_rows) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"  // row in the parent matrix
  "  const UWORD col = get_global_id(1); \n"  // col in the parent matrix
  "  if( (row <= end_row) && (col <= end_col) ) \n"
  "    { out[row + col*n_rows] /= val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,inplace_plus_array)(__global eT* out, __global const eT* A, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] += A[i]; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,inplace_minus_array)(__global eT* out, __global const eT* A, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] -= A[i]; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,inplace_mul_array)(__global eT* out, __global const eT* A, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] *= A[i]; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,inplace_div_array)(__global eT* out, __global const eT* A, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] /= A[i]; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_inplace_set_mat)(__global eT* out, __global const eT* A, const UWORD out_start_row, const UWORD out_start_col, const UWORD out_n_rows, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"  // row in source matrix
  "  const UWORD col = get_global_id(1); \n"  // col in source matrix
  "  if( (row <= A_n_rows) && (col <= A_n_cols) ) \n"
  "    { \n"
  "    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows); \n"
  "    const UWORD   A_index = row + col*A_n_rows; \n"
  "    out[out_index] = A[A_index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_inplace_plus_mat)(__global eT* out, __global const eT* A, const UWORD out_start_row, const UWORD out_start_col, const UWORD out_n_rows, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"  // row in source matrix
  "  const UWORD col = get_global_id(1); \n"  // col in source matrix
  "  if( (row <= A_n_rows) && (col <= A_n_cols) ) \n"
  "    { \n"
  "    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows); \n"
  "    const UWORD   A_index = row + col*A_n_rows; \n"
  "    out[out_index] += A[A_index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_inplace_minus_mat)(__global eT* out, __global const eT* A, const UWORD out_start_row, const UWORD out_start_col, const UWORD out_n_rows, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"  // row in source matrix
  "  const UWORD col = get_global_id(1); \n"  // col in source matrix
  "  if( (row <= A_n_rows) && (col <= A_n_cols) ) \n"
  "    { \n"
  "    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows); \n"
  "    const UWORD   A_index = row + col*A_n_rows; \n"
  "    out[out_index] -= A[A_index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_inplace_schur_mat)(__global eT* out, __global const eT* A, const UWORD out_start_row, const UWORD out_start_col, const UWORD out_n_rows, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"  // row in source matrix
  "  const UWORD col = get_global_id(1); \n"  // col in source matrix
  "  if( (row <= A_n_rows) && (col <= A_n_cols) ) \n"
  "    { \n"
  "    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows); \n"
  "    const UWORD   A_index = row + col*A_n_rows; \n"
  "    out[out_index] *= A[A_index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_inplace_div_mat)(__global eT* out, __global const eT* A, const UWORD out_start_row, const UWORD out_start_col, const UWORD out_n_rows, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"  // row in source matrix
  "  const UWORD col = get_global_id(1); \n"  // col in source matrix
  "  if( (row <= A_n_rows) && (col <= A_n_cols) ) \n"
  "    { \n"
  "    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows); \n"
  "    const UWORD   A_index = row + col*A_n_rows; \n"
  "    out[out_index] /= A[A_index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_plus_scalar)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = A[i] + val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_neg)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  (void)(val); \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = -(A[i]); } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_minus_scalar_pre)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = val - A[i]; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_minus_scalar_post)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = A[i] - val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_mul_scalar)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = A[i] * val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_div_scalar_pre)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = val / A[i]; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_div_scalar_post)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = A[i] / val; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_square)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  (void)(val); \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { const eT Ai = A[i]; out[i] = Ai * Ai; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_sqrt)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  (void)(val); \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = (eT)sqrt( (promoted_eT)(A[i]) ); } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_exp)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  (void)(val); \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = (eT)exp( (promoted_eT)(A[i]) ); } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_log)(__global eT* out, __global const eT* A, const eT val, const UWORD N) \n"
  "  { \n"
  "  (void)(val); \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = (eT)log( (promoted_eT)(A[i]) ); } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_plus_array)(__global eT* out, __global const eT* A, __global const eT* B, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = A[i] + B[i]; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_minus_array)(__global eT* out, __global const eT* A, __global const eT* B, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = A[i] - B[i]; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_mul_array)(__global eT* out, __global const eT* A, __global const eT* B, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = A[i] * B[i]; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,equ_array_div_array)(__global eT* out, __global const eT* A, __global const eT* B, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N)  { out[i] = A[i] / B[i]; } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,inplace_set_eye)(__global eT* out, const UWORD n_rows, const UWORD n_cols) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"
  "  const UWORD col = get_global_id(1); \n"
  "  if( (row < n_rows) && (col < n_cols) ) \n"
  "    { \n"
  "    const UWORD offset = row + col*n_rows; \n"
  "    out[offset] = (row == col) ? (eT)(1) : (eT)(0); \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,get_diag)(__global eT* out, __global const eT* A, const UWORD n_rows, const UWORD row_offset, const UWORD col_offset, const UWORD N) \n"
  "  { \n"
  "  const UWORD i = get_global_id(0); \n"
  "  if(i < N) \n"
  "    { \n"
  "    const UWORD index = (i + row_offset) + (i + col_offset)*n_rows; \n"
  "    out[i] = A[index]; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,trace)(__global eT* out, __global const eT* A, const UWORD n_rows, const UWORD N) \n"
  "  { \n"
  "  const UWORD id = get_global_id(0); \n"
  "  if(id == 0) \n"
  "    { \n"
  "    eT acc = (eT)(0); \n"
  "    #pragma unroll \n"
  "    for(UWORD i=0; i<N; ++i) \n"
  "      { \n"
  "      acc += A[i + i*n_rows];  \n"
  "      } \n"
  "    out[0] = acc; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,accu_simple)(__global eT* out, __global const eT* A, const UWORD A_len) \n"
  "  { \n"
  "  const UWORD id = get_global_id(0); \n"
  "  if(id == 0) \n"
  "    { \n"
  "    eT acc = (eT)(0); \n"
  "    #pragma unroll \n"
  "    for(UWORD i=0; i<A_len; ++i) \n"
  "      { acc += A[i]; } \n"
  "    out[0] = acc; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,accu_chunked)(__global eT* out, __global const eT* A, const UWORD chunk_size, const UWORD n_chunks) \n"
  "  { \n"
  "  const UWORD chunk_id = get_global_id(0); \n"
  "  if(chunk_id < n_chunks) \n"
  "    { \n"
  "    __global const eT* ptr = &(A[ chunk_id*chunk_size ]); \n"
  "    eT acc = (eT)(0); \n"
  "    #pragma unroll \n"
  "    for(UWORD i=0; i<chunk_size; ++i) \n"
  "      { acc += ptr[i]; } \n"
  "    out[chunk_id] = acc; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,accu_twostage)(__global eT* out, const UWORD out_len, __global const eT* A, const UWORD A_start, const UWORD A_len) \n"
  "  { \n"
  "  const UWORD id = get_global_id(0); \n"
  "  if(id == 0) \n"
  "    { \n"
  "    eT acc1 = (eT)(0); \n"
  "    #pragma unroll \n"
  "    for(UWORD i=A_start; i<A_len; ++i) \n"
  "      { acc1 += A[i]; } \n"
  "    \n"
  "    eT acc2 = (eT)(0); \n"
  "    #pragma unroll \n"
  "    for(UWORD i=0; i<out_len; ++i) \n"
  "      { acc2 += out[i]; } \n"
  "    \n"
  "    out[0] = acc1 + acc2; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,sum_colwise)(__global eT* out, __global const eT* A, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD col = get_global_id(0); \n"
  "  if(col < A_n_cols) \n"
  "    { \n"
  "    __global const eT* colptr = &(A[ col*A_n_rows ]); \n"
  "    eT acc = (eT)(0); \n"
  "    #pragma unroll \n"
  "    for(UWORD i=0; i < A_n_rows; ++i) \n"
  "      { acc += colptr[i]; } \n"
  "    out[col] = acc; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,sum_rowwise)(__global eT* out, __global const eT* A, const UWORD A_n_rows, const UWORD A_n_cols) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"
  "  if(row < A_n_rows) \n"
  "    { \n"
  "    eT acc = (eT)(0); \n"
  "    #pragma unroll \n"
  "    for(UWORD i=0; i < A_n_cols; ++i) \n"
  "      { acc += A[i*A_n_rows + row]; } \n"
  "    out[row] = acc; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_sum_colwise)(__global eT* out, __global const eT* A, const UWORD A_n_rows, const UWORD start_row, const UWORD start_col, const UWORD sub_n_rows, const UWORD sub_n_cols) \n"
  "  { \n"
  "  const UWORD col = get_global_id(0); \n"
  "  if(col < sub_n_cols) \n"
  "    { \n"
  "    __global const eT* colptr = &(A[ (col + start_col)*A_n_rows + start_row ]); \n"
  "    eT acc = (eT)(0); \n"
  "    #pragma unroll \n"
  "    for(UWORD i=0; i < sub_n_rows; ++i) \n"
  "      { acc += colptr[i]; } \n"
  "    out[col] = acc; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,submat_sum_rowwise)(__global eT* out, __global const eT* A, const UWORD A_n_rows, const UWORD start_row, const UWORD start_col, const UWORD sub_n_rows, const UWORD sub_n_cols) \n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"
  "  if(row < sub_n_rows) \n"
  "    { \n"
  "    eT acc = (eT)(0); \n"
  "    #pragma unroll \n"
  "    for(UWORD i=0; i < sub_n_cols; ++i) \n"
  "      { acc += A[(i+start_col)*A_n_rows + (row+start_row)]; } \n"
  "    out[row] = acc; \n"
  "    } \n"
  "  } \n"
  "\n"
  "__kernel void COOT_FN(PREFIX,ltri_set_zero)(__global eT* out, const UWORD n_rows, const UWORD n_cols)\n"
  "  { \n"
  "  const UWORD row = get_global_id(0); \n"
  "  const UWORD col = get_global_id(1); \n"
  "  const UWORD index = row + n_rows * col; \n"
  "  if( (row < n_rows) && (col < n_cols) && (row > col) ) \n"
  "    { \n"
  "    out[index] = (eT)(0); \n"
  "    } \n"
  "  }\n"
  ;
  
  return source;
  }
