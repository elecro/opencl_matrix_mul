SRC := $(shell find . -name "*.cpp")
OBJS  := $(patsubst %.cpp, %.o, $(SRC))

CXXFLAGS := -g -std=c++11

all: cpp_matrix_mul

cpp_matrix_mul: $(OBJS)
	$(CXX) -o $@ $(addprefix out/, $(OBJS)) -lOpenCL

%.o: %.cpp | out
	$(CXX) $(CXXFLAGS) -o out/$@  -c $<

out:
	mkdir out

clean:
	rm -f cpp_matrix_mul
