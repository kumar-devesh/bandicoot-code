// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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


// Utilities to get an unsigned integer type of the same width as the given type.




template<typename T> struct uint_type { };

template<> struct uint_type<u8>     { typedef u8 result;  };
template<> struct uint_type<s8>     { typedef u8 result;  };

template<> struct uint_type<u16>    { typedef u16 result; };
template<> struct uint_type<s16>    { typedef u16 result; };

template<> struct uint_type<float>  { typedef u32 result; };
template<> struct uint_type<u32>    { typedef u32 result; };
template<> struct uint_type<s32>    { typedef u32 result; };

template<> struct uint_type<double> { typedef u64 result; };
template<> struct uint_type<u64>    { typedef u64 result; };
template<> struct uint_type<s64>    { typedef u64 result; };

template<> struct uint_type<uword>  { typedef uword result; };

// Used sometimes by the kernel generation utilities to avoid specifying an unnecessary type.
template<> struct uint_type<void>   { typedef void result; };
