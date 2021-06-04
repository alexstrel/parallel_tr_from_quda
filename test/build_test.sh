nvcc -O0 -g -std=c++17 -I../build/include -I../build/include/targets/cuda -o transform_reduce_test.o -c transform_reduce_test.cu
nvcc -o transform_reduce_test.exe transform_reduce_test.o -L/home/astrel/Work/TR/transform_reduce_quda_v6/build/lib -lquda -lmpi

#nvcc -O3 -std=c++17 -I../build/include -I../build/include/targets/cuda -o transform_reduce_test.o -c transform_reduce_test.cpp

