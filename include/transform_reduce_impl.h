#pragma once
#include <reduce_helper.h>
#include <transform_reduce.h>
#include <tunable_reduction.h>
#include <kernels/transform_reduce.cuh>

namespace quda
{

  template <typename reduce_t, typename T, typename count_t, typename transformer, typename reducer>
  class TransformReduce : TunableMultiReduction<1>
  {
    using Arg = TransformReduceArg<reduce_t, T, count_t, transformer, reducer>;
    QudaFieldLocation location;
    std::vector<reduce_t> &result;
    const std::vector<T *> &v;
    count_t n_items;
    transformer &h;
    reduce_t init;
    reducer &r;

    bool tuneSharedBytes() const { return false; }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.grid.y = v.size();
    }

  public:
    TransformReduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v, count_t n_items,
                    transformer &h, reduce_t init, reducer &r) :
      TunableMultiReduction(n_items, v.size(), location),
      location(location),
      result(result),
      v(v),
      n_items(n_items),
      h(h),
      init(init),
      r(r)
    {
      strcpy(aux, "batch_size=");
      u32toa(aux + 11, v.size());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Arg arg(v, n_items, h, init, r);
      launch<transform_reducer, true>(result, tp, stream, arg);
    }

    long long bytes() const { return v.size() * n_items * sizeof(T); }
  };

  template <typename reduce_t, typename T, typename count_t, typename transformer, typename reducer>
  void transform_reduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v, count_t n_items,
                        transformer h, reduce_t init, reducer r)
  {
    if (result.size() != v.size()) errorQuda("result %lu and input %lu set sizes do not match", result.size(), v.size());
    TransformReduce<reduce_t, T, count_t, transformer, reducer> reduce(location, result, v, n_items, h, init, r);
  }

  template <typename reduce_t, typename T, typename count_t, typename transformer, typename reducer>
  reduce_t transform_reduce(QudaFieldLocation location, const T *v, count_t n_items, transformer h, reduce_t init, reducer r)
  {
    std::vector<reduce_t> result = {0.0};
    std::vector<const T *> v_ = {v};
    transform_reduce(location, result, v_, n_items, h, init, r);
    return result[0];
  }

  template <typename reduce_t, typename T, typename count_t, typename transformer, typename reducer>
  void reduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v, count_t n_items,
              reduce_t init, reducer r)
  {
    //transform_reduce(location, result, v, n_items, identity<T>(), init, r);
  }

  template <typename reduce_t, typename T, typename count_t, typename reducer>
  reduce_t reduce(QudaFieldLocation location, const T *v, count_t n_items, reduce_t init, reducer r)
  {
    std::vector<reduce_t> result = {0.0};
    std::vector<const T *> v_ = {v};
    //transform_reduce(location, result, v_, n_items, identity<T>(), init, r);
    return result[0];
  }
} // namespace quda
