
#LIB_FLAGS = -lbandicoot
#LIB_FLAGS = -lbandicoot -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda
LIB_FLAGS = -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda -larmadillo -lcurand -lcusolver -llapack -lblas

CXX_FLAGS = -std=c++11 -Wshadow -Wall -pedantic -g -I../include/ -O0 -DCOOT_USE_U64S64 -DCOOT_DEFAULT_BACKEND=CUDA_BACKEND
#CXX_FLAGS = -std=c++11 -Wshadow -Wall -pedantic -I../include/ -DCOOT_NO_DEBUG -O3 -DCOOT_USE_U64S64

OBJECTS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

%.o: %.cpp $(DEPS)
	$(CXX) $(CXX_FLAGS) -o $@ -c $<

%: %.o
	$(CXX) $(CXX_FLAGS) -o $@ $< $(LIB_FLAGS)

all: $(OBJECTS) matmul matvec chol accu fill copy submat_copy

.PHONY: clean

clean:
	rm -f matmul matvec chol accu fill copy submat_copy *.o
