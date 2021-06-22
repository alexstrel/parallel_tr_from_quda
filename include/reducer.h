#pragma once

// base reduction operations
namespace quda {

  template <typename T> struct plus {
    static constexpr bool do_sum = true;
    __device__ __host__ T operator()(T a, T b) const { return a + b; }
  };

  template <typename T> struct maximum {
    static constexpr bool do_sum = false;
    __device__ __host__ T operator()(T a, T b) const { return a > b ? a : b; }
  };

  template <typename T> struct minimum {
    static constexpr bool do_sum = false;
    __device__ __host__ T operator()(T a, T b) const { return a < b ? a : b; }
  };

  template <typename T> struct identity {
    static constexpr bool do_sum = false;
    __device__ __host__ T operator()(T a) const { return a; }
  };

}
