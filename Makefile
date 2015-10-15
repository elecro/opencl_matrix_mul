

all: matrix_mul

matrix_mul: main.c matrix_mul.cl
	$(CC) -o $@ main.c -lOpenCL

clean:
	rm -f matrix_mul
