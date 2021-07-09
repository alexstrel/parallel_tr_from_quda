#pragma once

//custom transform operations (DSL programming)

namespace quda {
  template <typename T> struct identity {
    static constexpr bool do_sum = false;
    T *x;
    identity(T *x_) : x(x_) {}
    T operator()(int i, int j = 0) const { return x[i]; }
  };
  
  template <typename T> struct axpyDot {
    static constexpr bool do_sum = false;  
    const T *x;
    T *y;
    const T a;

    axpyDot(const T a_, const T *x_, T *y_) : a(a_), x(x_), y(y_) {}

    T operator() (int i, int j = 0) const {
      y[i] = a*x[i] + y[i];
      return (y[i]*x[i]);
    }
  };  
}
