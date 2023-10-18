#LIB_FLAGS = -lbandicoot
#LIB_FLAGS = -lbandicoot -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda
LIB_FLAGS = -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda -lcurand -lcusolver -llapack -lblas -larmadillo

# NOTE: no wrapper in use!
CXX_FLAGS = -std=c++11 -Wshadow -Wall -pedantic -O0 -I../../include/ -g -DCOOT_USE_U64S64
BACKEND = CL_BACKEND
#BACKEND = CUDA_BACKEND

OBJECTS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

# Slight abuse on the object file extension here for convenience...
LAPACKTEST_OBJECTS = $(patsubst %.f,%.fo,$(wildcard *.f))

%.o: %.cpp $(DEPS)
	$(CXX) $(CXX_FLAGS) -DCOOT_DEFAULT_BACKEND=$(BACKEND) -o $@ -c $<
#	$(CXX) $(CXX_FLAGS) -o $@ -c $<

%.fo: %.f $(DEPS)
	$(FC) -o $@ -c $<

main: $(OBJECTS) $(LAPACKTEST_OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $(OBJECTS) $(LAPACKTEST_OBJECTS) $(LIB_FLAGS)


all: main

.PHONY: clean

clean:
	rm -f main *.o *.fo
