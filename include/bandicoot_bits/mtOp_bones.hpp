// Copyright 2020 Ryan Curtin (https://www.ratml.org/)
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



template<typename out_eT, typename T1, typename mtop_type>
class mtOp
  : public Base< out_eT, mtOp<out_eT, T1, mtop_type> >
  , public Op_traits<T1, mtop_type, has_nested_op_traits<mtop_type>::value>
  {
  public:

  typedef out_eT                                   elem_type;
  typedef typename T1::elem_type                   in_elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  static const bool is_row = T1::is_row;
  static const bool is_col = T1::is_col;

  const T1& q;

  inline         ~mtOp();
  inline explicit mtOp(const T1& in_m);
  };
