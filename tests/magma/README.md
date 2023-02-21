These tests are tests adapted from the MAGMA source code for the adaptations of
MAGMA functions for bandicoot's OpenCL backend.

Each test is individually ported from `magma-x.y.z/testing/testing_<func>.cpp`,
with irrelevant parts removed.  The general idea is to check the basic
functionality, instead of providing all of the options of the general testing
harness that MAGMA provides.

These functions depend on some LAPACK testing functionality that is not
distributed with standard LAPACK; therefore, we include the FORTRAN sources of
those test functions here.  They will also be compiled when the test is built.

These LAPACK test functions are licensed under the modified BSD license; more
details found here:

  https://netlib.org/lapack/LICENSE.txt
