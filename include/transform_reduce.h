#pragma once

#include <vector>
#include <enum_quda.h>
#include <complex_quda.h>

/**
   @file transform_reduce.h

   @brief QUDA reimplementation of thrust::transform_reduce as well as
   wrappers also implementing thrust::reduce.
 */

namespace quda
{
  namespace reducer {
    void init();
    void destroy();
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
  template <typename policy_t, typename reduce_t, typename count_t, typename reducer, typename transformer>
  reduce_t transform_reduce(policy_t &policy, count_t begin_it, count_t end_it, reduce_t init, reducer r, transformer h);
} // namespace quda
