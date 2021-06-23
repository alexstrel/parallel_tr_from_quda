#pragma once

//custom reduction operations (DSL programming)

namespace quda {
  template <typename T> struct identity {
    static constexpr bool do_sum = false;
    __device__ __host__ T operator()(T a) const { return a; }
  };
}
