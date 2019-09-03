
#LIB_FLAGS = -lbandicoot
#LIB_FLAGS = -lbandicoot -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda
LIB_FLAGS = -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda -larmadillo

#CXX_FLAGS = -std=c++11 -Wshadow -Wall -pedantic -O3 -DCOOT_NO_DEBUG -I../include/
CXX_FLAGS = -std=c++11 -Wshadow -Wall -pedantic -O0 -g -I../include/

OBJECTS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

%.o: %.cpp $(DEPS)
	$(CXX) -std=c++11 -Wshadow -Wall -pedantic -O3 -DCOOT_NO_DEBUG -I../include/ -o $@ -c $<

%: %.o
	$(CXX) $(CXX_FLAGS) -o $@ $< $(LIB_FLAGS)

all: matmul

.PHONY: clean

clean:
	rm -f matmul *.o
