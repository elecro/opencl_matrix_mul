SRC := $(shell find . -name "*.cpp")
OBJS  := $(patsubst %.cpp, %.o, $(SRC))

CXXFLAGS := -g -std=c++11

all: matrix_mul

matrix_mul: main.c matrix_mul.cl
	$(CC) -o $@ main.c -lOpenCL

cpp_matrix_mul: $(OBJS)
	$(CXX) -o $@ $(addprefix out/, $(OBJS)) -lOpenCL

%.o: %.cpp | out
	$(CXX) $(CXXFLAGS) -o out/$@  -c $<

out:
	mkdir out

clean:
	rm -f matrix_mul
