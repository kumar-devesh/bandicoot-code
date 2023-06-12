
# Use this when compiling without CUDA.
# (If you want to do that, also comment out COOT_USE_CUDA in bandicoot_bits/config.hpp.)
#LIB_FLAGS = -lOpenCL -lclBLAS -larmadillo -llapack -lblas
LIB_FLAGS = -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda -lcurand -lcusolver -llapack -lblas -larmadillo

CXX_FLAGS = -std=c++11 -Wshadow -Wall -pedantic -O0 -I../include/ -g -DCOOT_USE_U64S64
# To compile these tests, select a backend by uncommenting one of the following
# lines.
#BACKEND = CL_BACKEND
#BACKEND = CUDA_BACKEND

OBJECTS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

%.o: %.cpp $(DEPS)
	$(CXX) $(CXX_FLAGS) -DCOOT_DEFAULT_BACKEND=$(BACKEND) -o $@ -c $<
#	$(CXX) $(CXX_FLAGS) -o $@ -c $<

main: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $(OBJECTS) $(LIB_FLAGS)


all: main

.PHONY: clean

clean:
	rm -f main *.o
