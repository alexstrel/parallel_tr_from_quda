#pragma once

//custom transform operations (DSL programming)

namespace quda {
  template <typename T> struct identity {
    static constexpr bool do_sum = false;
    T *data;
    __device__ __host__ identity(T *data_) : data(data_) {}
    __device__ __host__ T operator()(int i, int j) const { return data[i]; }
  };
}
