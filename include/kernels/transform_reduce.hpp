#pragma once

#include <reduction_kernel.h>
#include <limits>

namespace quda {


  template <typename reduce_t, int n_batch_, typename reducer_, typename transformer_>
  struct TransformReduceArg : public ReduceArg<reduce_t> {
    using reducer = reducer_;
    using transformer = transformer_;
    static constexpr int n_batch_max = 8;
    int n_items;
    int n_batch;
    reduce_t init_value;
    reducer r;
    transformer h;

    TransformReduceArg(int n_items, reduce_t init_value, reducer r, transformer h) :
      ReduceArg<reduce_t>(n_batch_),
      n_items(n_items),
      n_batch(n_batch_),
      init_value(init_value),
      r(r),
      h(h)      
    {
      if (n_batch > n_batch_max) errorQuda("Requested batch %d greater than max supported %d", n_batch, n_batch_max);
      if (n_items > std::numeric_limits<int>::max())
        errorQuda("Requested size %lu greater than max supported %lu",
                  (uint64_t)n_items, (uint64_t)std::numeric_limits<int>::max());
      this->threads = dim3(n_items, n_batch, 1);
    }

    reduce_t init() const { return init_value; }
  };

  template <typename Arg> struct transform_reducer {
    using count_t = decltype(Arg::n_items);
    using reduce_t = decltype(Arg::init_value);

    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr transform_reducer(const Arg &arg) : arg(arg) {}

    static constexpr bool do_sum = Arg::reducer::do_sum;

    inline reduce_t operator()(reduce_t a, reduce_t b) const { return arg.r(a, b); }

    inline reduce_t operator()(reduce_t &value, count_t i, int j, int)//j is a batch indx
    {
      auto t = arg.h(i, j);
      return arg.r(t, value);
    }
  };

}
