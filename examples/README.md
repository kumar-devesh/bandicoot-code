# How to compile example1.cpp

## Linux and Mac OS

If you have installed Bandicoot via the CMake installer:

```
g++ example1.cpp -o example1 -std=c++11 -O2 -lbandicoot
```

Otherwise, if you want to use Bandicoot without installation, first modify `include/bandicoot_bits/config.hpp` to match your system configuration (see https://coot.sourceforge.io/docs.html#config_hpp), then:

```
g++ example1.cpp -o example1 -std=c++11 -O2 -I /home/user/bandicoot-1.0.0/include/ -DCOOT_DONT_USE_WRAPPER -lcuda -lcudart -lcublas -lcurand -lcusolver -lnvrtc -lOpenCL -lclBLAS
```

 * If CUDA is not available, omit `-lcuda -lcudart -lcublas -lcurand -lcusolver -lnvrtc`
 * If OpenCL is not available, omit `-lOpenCL -lclBLAS`
 * If using Mac OS, use `-framework OpenCL` instead of `-lOpenCL`

## Windows

Bandicoot is still in early development and has not been tested on Windows.
If you are willing to try and contribute a working example solution, it would be welcomed!
(Contribute to https://gitlab.com/conradsnicta/bandicoot-code/)
