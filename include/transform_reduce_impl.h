#pragma once
#include <reduce_helper.h>
#include <transform_reduce.h>
#include <tunable_reduction.h>
#include <kernels/transform_reduce.cuh>
#include <array>

namespace quda
{

  template <typename policy_t, typename reduce_t, int n_batch_, typename reducer, typename transformer>
  class TransformReduce : TunableMultiReduction<1>
  {
    using Arg = TransformReduceArg<reduce_t, n_batch_, reducer, transformer>;
    
    policy_t policy;
    std::vector<reduce_t> &result;
    int n_items;
    reduce_t init;
    reducer r;
    transformer h;        

    bool tuneSharedBytes() const { return false; }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.grid.y = n_batch_;
    }

  public:
  
    TransformReduce(policy_t &policy, std::vector<reduce_t> &result, int n_items, reduce_t init, reducer r, transformer h) :
      TunableMultiReduction(n_items, n_batch_, policy),//policy keeps location
      policy(policy),
      result(result),
      n_items(n_items),
      init(init),
      r(r),
      h(h)      
    {
      strcpy(aux, "batch_size=");
      u32toa(aux + 11, n_batch_);
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      //
      Arg arg(n_items, init, r, h);
      launch<transform_reducer, true>(result, tp, stream, arg);
    }

    long long bytes() const { return nbatch * n_items * sizeof(reduce_t); }//need to deduce from h
  };

  template <typename policy_t, typename reduce_t, typename count_t, typename reducer, typename transformer>
  reduce_t transform_reduce(policy_t &policy, count_t begin_it, count_t end_it, reduce_t init, reducer r, transformer h)
  {
    constexpr int n_batch = 1;
    std::vector<reduce_t> result = {0.0};
    const int n_items = end_it - begin_it;

    TransformReduce<policy_t, reduce_t, n_batch, reducer, transformer> transformReducer(policy, result, n_items, init, r, h);
    
    //if constexpr (!is_async) policy.get_queue().wait();
    
    return result[0];
  }  

} // namespace quda
