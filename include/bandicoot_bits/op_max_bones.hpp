// Copyright 2021 Ryan Curtin (https://www.ratml.org/)
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


//! \addtogroup op_max
//! @{


class op_max
  {
  public:

  template<typename T1>
  inline static typename T1::elem_type apply(const Op<T1, op_max>& in);

  template<typename T1>
  inline static typename T1::elem_type apply(const Op<eOp<T1, eop_abs>, op_max>& in);
  };



//! @}