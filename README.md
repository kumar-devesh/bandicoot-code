### Bandicoot: C++ Library for GPU Linear Algebra & Scientific Computing
https://coot.sourceforge.io

Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)  
Copyright 2017-2023 Marcus Edel (https://kurg.org)  
Copyright 2017-2023 Conrad Sanderson (https://conradsanderson.id.au)

---

### Quick Links

- [download latest stable release](https://coot.sourceforge.io/download.html)
- [documentation for functions and classes](https://coot.sourceforge.io/docs.html)
- [bug reports & questions](https://coot.sourceforge.io/faq.html)

### Contents

1.  [Introduction](#1-introduction)
2.  [Citation Details](#2-citation-details)
3.  [Distribution License](#3-distribution-license)

4.  [Prerequisites and Dependencies](#4-prerequisites-and-dependencies)

5.  [Linux and macOS: Installation](#5-linux-and-macos-installation)
6.  [Linux and macOS: Compiling and Linking](#6-linux-and-macos-compiling-and-linking)

7.  [Windows Support](#7-windows-support)

8.  [Adapting Armadillo Code to Bandicoot](#8-adapting-armadillo-code-to-bandicoot)
9.  [Caveat on use of C++11 auto Keyword](#9-caveat-on-use-of-c11-auto-keyword)
10. [Support for OpenMP](#11-support-for-openmp)

11. [Documentation of Functions and Classes](#11-documentation-of-functions-and-classes)
12. [API Stability and Version Policy](#12-api-stability-and-version-policy)
13. [Bug Reports and Frequently Asked Questions](#13-bug-reports-and-frequently-asked-questions)

---

### 1. Introduction

Bandicoot is a high quality C++ library for GPU linear algebra and scientific computing,
aiming towards a good balance between speed and ease of use.
It supports the use of CUDA and OpenCL devices.

It's useful for GPU algorithm development directly in C++,
and/or quick conversion of research code into GPU-enabled production environments.
It has high-level syntax and functionality which is deliberately similar to Matlab,
and mirrors the API of the CPU-based [Armadillo](https://arma.sourceforge.net/)
C++ linear algebra library.

Bandicoot provides efficient classes for vectors and matrices,
as well as many associated functions covering essential
and advanced functionality for data processing and manipulation of matrices.

Various matrix decompositions (eigen, SVD, etc.) are provided through GPU implementations,
either internal to Bandicoot or through external libraries such as cuSolver.

A sophisticated expression evaluator (via C++ template meta-programming)
automatically combines several operations (at compile time) to increase speed and efficiency.

The library can be used for machine learning, pattern recognition, computer vision,
signal processing, bioinformatics, statistics, finance, etc.

Authors:
  * Ryan Curtin      - https://www.ratml.org
  * Marcus Edel      - https://kurg.org
  * Conrad Sanderson - https://conradsanderson.id.au

---

### 2: Citation Details

Please cite the following papers if you use Bandicoot in your research and/or software.
Citations are useful for the continued development and maintenance of the library.

  * TODO!

---

### 3: Distribution License

Bandicoot can be used in both open-source and proprietary (closed-source) software.

Bandicoot is licensed under the Apache license, Version 2.0 (the "License").
A copy of the License is included in the "LICENSE.txt" file.

Any software that incorporates or distributes Bandicoot in source or binary form must include,
in the documentation and/or other materials provided with the software,
a readable copy of the attribution notices present in the "NOTICE.txt" file.
See the License for details. The contents of the "NOTICE.txt" file are for
informational purposes only and do not modify the License.

----

### 4: Prerequisites and Dependencies

The functionality of Bandicoot is partly dependent on other libraries:
- OpenBLAS (or standard BLAS)
- LAPACK
- *CUDA backend*: cuda, cudart, cuBLAS, cuRand, cuSolver (the CUDA toolkit)
- *OpenCL backend*: OpenCL, clBLAS

Bandicoot requires at least one GPU backend to function.
Currently, CUDA and OpenCL are supported.

Use of OpenBLAS (instead of standard BLAS) is strongly recommended on all systems.
On macOS, the Accelerate framework can be used for BLAS and LAPACK functions.
BLAS and LAPACK are used in some decompositions for mixed-mode CPU/GPU computation.

Bandicoot requires a C++ compiler that supports at least the C++11 standard.

On Linux-based systems, install the GCC C++ compiler, which is available as a pre-built package.
The package name might be `g++` or `gcc-c++` depending on your system.

On macOS systems, a C++ compiler can be obtained by first installing Xcode (at least version 8)
and then running the following command in a terminal window:

```
xcode-select --install
```

On Windows sytems, the MinGW toolset or Visual Studio C++ 2019 (MSVC) can be used.

---

### 5: Linux and macOS: Installation

Bandicoot can be installed in several ways: either manually or via CMake, with or without root access.
The CMake-based installation is preferred.
CMake can be downloaded from https://www.cmake.org
or (preferably) installed using the package manager on your system;
on macOS systems, CMake can be installed through MacPorts or Homebrew.

Before installing Bandicoot, first install OpenBLAS and LAPACK,
and the following dependencies for the backend(s) you would like to use:

 * *OpenCL*: OpenCL and clBLAS
 * *CUDA*: the CUDA toolkit including CUDA, cudart, cuRand, cuBLAS, cuSolver

For each of these dependencies, it is also necessary to install the corresponding development files for each library.
For example, when installing the `libopenblas` package, also install the `libopenblas-dev` package.

#### 5a: Installation via CMake

The CMake-based installer detects which relevant libraries
are installed on your system (e.g. OpenBLAS, LAPACK, OpenCL, CUDA, etc.)
and correspondingly modifies Bandicoot's configuration.
The installer also generates the Bandicoot runtime library,
which is a wrapper for all the detected libraries.

Change into the directory that was created by unpacking the Bandicoot archive (e.g. `cd bandicoot-1.0.0`) and then run CMake using:

```
cmake .
```

**NOTE:** the full stop (`.`) separated from `cmake` by a space is important.

On macOS, to enable the detection of OpenBLAS,
use the additional `ALLOW_OPENBLAS_MACOS` option when running CMake:

```
cmake -DALLOW_OPENBLAS_MACOS=ON .
```

Depending on your installation, OpenBLAS may masquerade as standard BLAS.
To detect standard BLAS and LAPACK, use the `ALLOW_BLAS_LAPACK_MACOS1` option:

```
cmake -DALLOW_BLAS_LAPACK_MACOS=ON .
```

By default, CMake assumes that the Bandicoot runtime library and the corresponding header files
will be installed in the default system directory (e.g. in the `/usr` hierarchy on Linux-based systems).
To install the library and headers in an alternative directory,
use the additional option `CMAKE_INSTALL_PREFIX` in this form:

```
cmake -DCMAKE_INSTALL_PREFIX:PATH=alternative_directory .
```

If CMake needs to be re-run, it's a good idea to first delete the `CMakeCache.txt` file (not `CMakeLists.txt`).

**Caveat:** if Bandicoot is installed in a non-system directory,
make sure that the C++ compiler is configured to use the `lib` and `include`
sub-directories present within this directory.

Note that the `lib` directory might be named differently on your system.
On recent 64-bit Debian and Ubuntu systems it is `lib/x86_64-linux-gnu`.
On recent 64-bit Fedora and RHEL systems it is `lib64`.

If you have sudo access (i.e. root/administrator/superuser privileges)
and didn't use the `CMAKE_INSTALL_PREFIX` option, run the following command:

```
sudo make install
```

If you don't have sudo access, make sure to use the `CMAKE_INSTALL_PREFIX` option
and run the following command:

```
make install
```

#### 5b: Manual Installation

Manual installation involves simply copying the `include/bandicoot` header
**and** the associated `include/bandicoot_bits` directory
to a location such as `/usr/include/` which is searched by your C++ compiler.
If you don't have sudo access or don't have write access to `/usr/include/`,
use a directory within your own home directory (e.g. `/home/user/include/`).

If required, modify `include/bandicoot_bits/config.hpp`
to indicate which libraries are currently available on your system.
Comment or uncomment the following lines:

```
#define COOT_USE_OPENCL
#define COOT_USE_CUDA
```

Note that the manual installation will not generate the Bandicoot runtime library,
and hence you will need to link your programs directly with OpenBLAS, LAPACK, CUDA, OpenCL, etc.;
see the [direct linking](https://coot.sourceforge.io/docs.html#direct_linking)
section of the documentation for more details.

---

### 6: Linux and macOS: Compiling and Linking

If you have installed Bandicoot via the CMake installer,
use the following command to compile your program:

```
g++ prog.cpp -o prog -O2 -std=c++11 -lbandicoot
```

If you have installed Bandicoot manually, link with OpenBLAS and LAPACK
and the required libraries for the backend you are using,
instead of the Bandicoot runtime library.
If only the OpenCL backend is enabled
(i.e. only `COOT_USE_OPENCL` is defined in `config.hpp`),
use this command:

```
g++ prog.cpp -o prog -O2 -std=c++11 -lopenblas -llapack -lOpenCL -lclBLAS
```

If using only the CUDA backend (i.e. only `COOT_USE_CUDA` is defined in `config.hpp`), use this command:

```
g++ prog.cpp -o prog -O2 -std=c++11 -lopenblas -llapack -lcuda -lcudart -lcurand -lcublas -lcusolver
```

If both backends are enabled, use this command:

```
g++ prog.cpp -o prog -O2 -std=c++11 -lopenblas -llapack -lOpenCL -lclBLAS -lcuda -lcudart -lcurand -lcublas -lcusolver
```

If you have manually installed Bnadicoot in a non-standard location,
such as `/home/user/include/`, you will need to make sure
that your C++ compiler searches `/home/user/include/`
by explicitly specifying the directory as an argument/option.
For example, the `-I` flag can be used in GCC and Clang
(the example assumes only the OpenCL backend is enabled);

```
g++ prog.cpp -o prog -O2 -std=c++11 -I /home/user/include/ -lopenblas -llapack -lOpenCL -lclBLAS
```

If you're getting linking issues (unresolved symbols),
enable the `COOT_DONT_USE_WRAPPER` option
(the example assumes only the OpenCL backend is enabled):

```
g++ prog.cpp -o prog -O2 -std=c++11 -I /home/user/include/ -DCOOT_DONT_USE_WRAPPER -lopenblas -llapack -lOpenCL -lclBLAS
```

If you don't have OpenBLAS, on Linux change `-lopenblas` to `-lblas`;
on macOS change `-lopenblas -llapack` to `-framework Accelerate`.

The `examples/` directory contains a short example program that uses Bandicoot.

We recommend that compilation is done with optimisation enabled,
in order to make the best use of the template meta-programming
techniques employed in Bandicoot.
For GCC and Clang compilers, use `-O2` or `-O3` to enable optimisation.

For more information on compiling and linking, see the following resources:

 * https://coot.sourceforge.io/faq.html
 * https://coot.sourceforge.io/docs.html#direct_linking

---

### 7. Windows Support

Bandicoot has so far not been thoroughly tested on Windows.
Contributions to improve Windows support are appreciated.

---

### 8: Adapting Armadillo Code to Bandicoot

Users of Armadillo can adapt their code that runs on CPUs to Bandicoot.
Bandicoot aims to be API-compatible with Armadillo.
As a first step, replace `#include <armadillo>` with `#include <bandicoot>`,
and any uses of `arma::` (or `using namespace arma`) with `coot::` (or `using namespace coot`).

Due to inherent architectural differences between GPUs and CPUs, the following caveats apply: 

  * GPUs are best suited for operations on large matrices,
    so small matrices (e.g. with size ≤ 100x100) may not obtain speedups
  
  * Individual element access such as X(i,j) has an overhead of transferring between the GPU and CPU;
    when adapting Armadillo code to Bandicoot, direct element access should be avoided
    
  * Where possible, use batch operations with Bandicoot; e.g., use `A += 1` instead of
    `for(uword i=0; i<A.n_elem; ++i) { A[i] += 1; }`
  
  * If direct element access cannot be avoided, consider temporarily transferring
    the entire Bandicoot matrix to CPU-accessible memory by creating an Armadillo matrix
    via `conv_to<arma::fmat>(X)`
    
  * Due to the overhead of direct element access, Bandicoot does not provide iterators
  
  * Consumer-level GPUs typically obtain better performance with 32-bit floating point elements
    rather than 64-bit (e.g. `float` instead of `double`), so using `fmat` instead of `mat` is preferable

See also the Armadillo/Bandicoot conversion guide:
https://coot.sourgeforge.net/docs.html#arma_comparison

---

### 9: Caveat on use of C++11 auto Keyword

Use of the C++11 `auto` keyword is not recommended with Bandicoot objects and expressions.

Bandicoot has a template meta-programming framework that creates lots of short-lived temporaries
that are not properly handled by `auto`.

---

### 10: Support for OpenMP

Bandicoot can use OpenMP to parallelize some operations.
This requires a C++ compile with OpenMP 3.1+ support.

For GCC and Clang compilers, use the following option to enable OpenMP:
`-fopenmp`.

**Note:** because OpenMP parallelizes on the CPU and not the GPU,
observed speedup for enabling OpenMP may be minimal,
as it will only parallelize a few CPU-specific parts of Bandicoot algorithms.

---

### 11: Documentation of Functions and Classes

The documentation of Bandicoot functions and classes is available at:
https://coot.sourceforge.io/docs.html

The documentation is also in the `docs.html` file distributed with Bandicoot.
Use a web browser to view it.

---

### 12: API Stability and Version Policy

Each release of Bandicoot has its public API (functions, classes, constants)
described in the accompanying API documentation (docs.html) specific to that release.

Each release of Bandicoot has its full version specified as A.B.C,
where A is the major version number,
B is the minor version number,
and C is a patch level (indicating bug fixes).
The version specification has explicit meaning,
similar to [Semantic Versioning](https://semver.org/), as follows:

 * Within a major version (e.g. 1), each minor version has a public API that
   strongly strives to be backwards compatible (at the source level) with the
   public API of preceding minor versions.  For example, user code written for
   version 1.0 should work with version 1.1, 1.2, etc.;
   however, later minor versions may have more features (API additions and extensions)
   than preceding minor versions.  As such, use code _specifically_
   written for version 1.2 may not work with 1.1.

 * An increase in the patch level, while the major and minor versions are retained,
   indicates modifications to the code and/or documentation which aim to fix bugs
   without altering the public API.

 * We don't like changes to existing public API and strongly prefer not to break
   any user software.  However, to allow evolution, the public API in future major versions may change,
   while remaining backwards compatible in as many cases as possible
   (e.g. major version 2 may have slightly different public API than major version 1).

**CAVEAT:**
the above policy applies only to the public API described in the documentation.
Any functionality within Bandicoot which is _not explicitly_ described
in the public API documentation is considered to be internal implementation details,
and may be changed or removed without notice.

---

### 13: Bug Reports and Frequently Asked Questions

Bandicoot has gone through extensive testing.
However, as with almost all software, it's impossible to guarantee 100% correct functionality.

If you find a bug in the library or the documentation,
we are interested in hearing about it.
Please make a _small_ and _self-contained_ program which exposes the bug,
and then send the program source and the bug description to the developers.
The small program must have a `main()` function and use only functions/classes from Bandicoot,
the standard C++ library, and possibly Armadillo (no other libraries).

If you are adapting Armadillo code to Bandicoot
and find that Bandicoot is missing functionality,
we are also interested in hearing about it,
so that development efforts can be prioritized.

The contact details for bug reporting can be found at:
https://coot.sourceforge.io/contact.html

Further information about Bandicoot is on the Frequently Asked Questions page:
https://coot.sourceforge.io/faq.html
