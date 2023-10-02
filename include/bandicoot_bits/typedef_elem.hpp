// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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


// we need our own typedefs for types to use in template code;
// OpenCL attaches attributes to cl_int, cl_long, ...,
// which can cause lots of "attribute ignored" warnings
// when such types are used in template code


#if defined(UINT8_MAX)
  typedef          uint8_t   u8;
  typedef           int8_t   s8;
#elif (UCHAR_MAX == 0xff)
  typedef unsigned char      u8;
  typedef   signed char      s8;
#else
  typedef          cl_uchar  u8;
  typedef          cl_char   s8;
#endif


#if defined(UINT16_MAX)
  typedef          uint16_t  u16;
  typedef           int16_t  s16;
#elif (USHRT_MAX == 0xffff)
  typedef unsigned short     u16;
  typedef          short     s16;
#else
  typedef          cl_ushort u16;
  typedef          cl_short  s16;
#endif


#if defined(UINT32_MAX)
  typedef          uint32_t u32;
  typedef           int32_t s32;
#elif (UINT_MAX == 0xffffffff)
  typedef unsigned int      u32;
  typedef          int      s32;
#else
  typedef          cl_uint  u32;
  typedef          cl_int   s32;
#endif


#if defined(UINT64_MAX)
  typedef          uint64_t  u64;
  typedef           int64_t  s64;
#elif (ULLONG_MAX == 0xffffffffffffffff)
  typedef unsigned long long u64;
  typedef          long long s64;
#elif (ULONG_MAX  == 0xffffffffffffffff)
  typedef unsigned long      u64;
  typedef          long      s64;
  #define COOT_U64_IS_LONG
#else
  typedef          cl_ulong  u64;
  typedef          cl_long   s64;
#endif


// need both signed and unsigned versions of size_t
typedef          std::size_t                         uword;
typedef typename std::make_signed<std::size_t>::type sword;


#if   defined(COOT_BLAS_LONG_LONG)
  typedef long long blas_int;
  #define COOT_MAX_BLAS_INT 0x7fffffffffffffffULL
#elif defined(COOT_BLAS_LONG)
  typedef long      blas_int;
  #define COOT_MAX_BLAS_INT 0x7fffffffffffffffUL
#else
  typedef int       blas_int;
  #define COOT_MAX_BLAS_INT 0x7fffffffU
#endif


typedef std::complex<float>  cx_float;
typedef std::complex<double> cx_double;

typedef void* void_ptr;
