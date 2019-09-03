
#LIB_FLAGS = -lbandicoot
#LIB_FLAGS = -lbandicoot -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda
LIB_FLAGS = -lOpenCL -lclBLAS -lnvrtc -lcudart -lcublas -lcuda

CXX_FLAGS = -std=c++11 -Wshadow -Wall -pedantic -O0 -I../include/ -g
BACKEND = USE_CL_BACKEND
#BACKEND = USE_CUDA_BACKEND

OBJECTS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

%.o: %.cpp $(DEPS)
	$(CXX) $(CXX_FLAGS) -D $(BACKEND) -o $@ -c $<

main: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $(OBJECTS) $(LIB_FLAGS)


all: main

.PHONY: clean

clean:
	rm -f main *.o
