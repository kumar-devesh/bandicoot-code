LIB_FLAGS = -lOpenCL -lclBLAS -llapack -lblas -larmadillo

# NOTE: no wrapper in use!
CXX_FLAGS = -std=c++11 -Wshadow -Wall -pedantic -O0 -I../../include/ -g -DCOOT_DEFAULT_BACKEND=CL_BACKEND -DCOOT_DONT_USE_CUDA -DCOOT_USE_OPENCL

OBJECTS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

# Slight abuse on the object file extension here for convenience...
LAPACKTEST_OBJECTS = $(patsubst %.f,%.fo,$(wildcard *.f))

%.o: %.cpp $(DEPS)
	$(CXX) $(CXX_FLAGS) -o $@ -c $<

%.fo: %.f $(DEPS)
	$(FC) -o $@ -c $<

main: $(OBJECTS) $(LAPACKTEST_OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $(OBJECTS) $(LAPACKTEST_OBJECTS) $(LIB_FLAGS)


all: main

.PHONY: clean

clean:
	rm -f main *.o *.fo
