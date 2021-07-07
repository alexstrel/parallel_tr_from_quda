#include <transform_reduce_impl.h>
//implementations:
#include <reducer.h>
#include <transformer.h>

namespace quda
{
  using iter_f32_t = decltype(std::vector<float, AlignedAllocator<float>>().begin());
  
  // explicit instantiation list for transform_reduce
  template float transform_reduce<QudaFieldLocation, float, int, plus<float>, identity<float>>(
    QudaFieldLocation&, int, int, float, plus<float>, identity<float>);

  template float transform_reduce<QudaFieldLocation, float, iter_f32_t, plus<float>, identity<float>>(
      QudaFieldLocation&, iter_f32_t, iter_f32_t, float, plus<float>, identity<float>);   
      
  template float transform_reduce<QudaFieldLocation, float, iter_f32_t, plus<float>, axpyDot<float>>(
      QudaFieldLocation&, iter_f32_t, iter_f32_t, float, plus<float>, axpyDot<float>);   
      
} // namespace quda
