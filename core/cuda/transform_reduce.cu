#include <transform_reduce_impl.h>
//implementations:
#include <reducer.h>
#include <transformer.h>

using iter_f32_t = decltype(std::vector<float, quda::AlignedAllocator<float>>().begin());

namespace quda
{
  // explicit instantiation list for transform_reduce
  template float transform_reduce<QudaFieldLocation, float, int, plus<float>, identity<float>>(
    QudaFieldLocation&, int, int, float, plus<float>, identity<float>);

  template float transform_reduce<QudaFieldLocation, float, iter_f32_t, plus<float>, identity<float>>(
      QudaFieldLocation&, iter_f32_t, iter_f32_t, float, plus<float>, identity<float>);   
} // namespace quda
