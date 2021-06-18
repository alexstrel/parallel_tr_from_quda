#pragma once

#include <vector>
#include <reduce_helper.h>
#include <enum_quda.h>
#include <complex_quda.h>
#include <reducer.h>
#include <tunable_reduction.h>
#include <kernels/transform_reduce.cuh>

namespace quda
{
  namespace reducer {
    void init();
    void destroy();
  }

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
  
  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory, and is a batched implementation.
     @param[in] location Location where the reduction will take place
     @param[out] result Vector of results
     @param[in] v Vector of inputs
     @param[in] n_items Number of elements to be reduced in each input
     @param[in] transformer Functor that applies transform to each element
     @param[in] init The results are initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename count_t, typename transformer, typename reducer>
  void transform_reduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v, count_t n_items,
                        transformer h, reduce_t init, reducer r)
  {
    if (result.size() != v.size()) errorQuda("result %lu and input %lu set sizes do not match", result.size(), v.size());
    TransformReduce<reduce_t, T, count_t, transformer, reducer> reduce(location, result, v, n_items, h, init, r);
  }

  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
     @param[in] transformer Functor that applies transform to each element
     @param[in] init Results is initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename count_t, typename transformer, typename reducer>
  reduce_t transform_reduce(QudaFieldLocation location, const T *v, count_t n_items, transformer h, reduce_t init, reducer r)
  {
    std::vector<reduce_t> result = {0.0};
    std::vector<const T *> v_ = {v};
    transform_reduce(location, result, v_, n_items, h, init, r);
    return result[0];
  }

  /**
     @brief QUDA implementation providing thrust::reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory, and is a batched implementation.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
     @param[in] init The results are initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename count_t, typename transformer, typename reducer>
  void reduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v, count_t n_items,
              reduce_t init, reducer r)
  {
    transform_reduce(location, result, v, n_items, identity<T>(), init, r);
  }

  /**
     @brief QUDA implementation providing thrust::reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
     @param[in] init Result is initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename count_t, typename reducer>
  reduce_t reduce(QudaFieldLocation location, const T *v, count_t n_items, reduce_t init, reducer r)
  {
    std::vector<reduce_t> result = {0.0};
    std::vector<const T *> v_ = {v};
    transform_reduce(location, result, v_, n_items, identity<T>(), init, r);
    return result[0];
  }

} // namespace quda
