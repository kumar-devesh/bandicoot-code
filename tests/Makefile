
#LIB_FLAGS = -lbandicoot
#LIB_FLAGS = -lbandicoot -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda
LIB_FLAGS = ./lapack-test.a -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda -lcurand -lcusolver -llapack -lblas -larmadillo

# Get lapack-test.a like this:
# Download the LAPACK sources.
# Build everything.
#
# cd TESTING/EIG/
# ar r lapack-test.a *.o
#
# And then just copy it to this directory.
# TODO: clean that up...

CXX_FLAGS = -std=c++11 -Wshadow -Wall -pedantic -O0 -I../include/ -g -DCOOT_USE_U64S64
BACKEND = CL_BACKEND
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
