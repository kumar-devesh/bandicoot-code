**About**
* Bandicoot aims:
   * GPU accelerator add-on for the [Armadillo](http://arma.sourceforge.net) C++ linear algebra library,
     ie. subset of functions (such as matrix decompositions) which process Armadillo matrices on GPUs
   * stand-alone linear algebra library for computing on GPUs, providing a subset of Armadillo functionality
* Bandicoot is currently a **work-in-progress** and hence currently only for experimental use
<br>
<br>

**Requirements** (subject to change)
* Armadillo 8.500 or later - http://arma.sourceforge.net
* OpenCL / CUDA
  - eg. AMDGPU-PRO (for AMD hardware), [ROCm](https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime) OpenCL runtime (for AMD hardware), [Neo](https://01.org/compute-runtime) (for Intel hardware), or CUDA (for NVIDIA hardware), or POCL
* clBLAS - https://github.com/clMathLibraries/clBLAS
* clBLast - https://github.com/CNugteren/CLBlast (tuned BLAS for GPUs)
<br>
<br>

**NOTES**
- do not use the Beignet OpenCL driver for Intel hardware; Beignet is horribly broken and no longer developed or maintained
- do not use the Mesa OpenCL (clover) driver; it's incomplete and full of bugs

**Authors**
* Conrad Sanderson - http://conradsanderson.id.au
* Ryan Curtin - http://www.ratml.org
<br>
<br>
