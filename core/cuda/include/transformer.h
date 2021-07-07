#pragma once

//custom transform operations (DSL programming)

namespace quda {
  template <typename T> struct identity {
    static constexpr bool do_sum = false;
    T *x;
    __device__ __host__ identity(T *x_) : x(x_) {}
    __device__ __host__ T operator()(int i, int j = 0) const { return x[i]; }
  };
  
  template <typename T> struct axpyDot {
    static constexpr bool do_sum = false;  
    const T *x;
    T *y;
    const T a;

    __device__ __host__  axpyDot(const T a_, const T *x_, T *y_) : a(a_), x(x_), y(y_) {}

    __device__ __host__ T operator() (int i, int j = 0) const {
      y[i] = a*x[i] + y[i];
      return (y[i]*x[i]);
    }
  };  
}
