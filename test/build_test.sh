nvcc -O0 -g -std=c++17 --expt-relaxed-constexpr -I../build/include -I../build/include/targets/cuda -I../core/cuda/include -o transform_reduce_test.o -c transform_reduce_test.cu
nvcc -o transform_reduce_test.exe transform_reduce_test.o -L/home/astrel/Work/TR/parallel_tr_from_quda_cpp_v3/build/lib -lquda -lmpi

