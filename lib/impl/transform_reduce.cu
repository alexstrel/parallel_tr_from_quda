#include <transform_reduce_impl.h>

namespace quda
{
  // explicit instantiation list for transform_reduce
  template float transform_reduce<float, float, int, identity<float>, plus<float>>(
    QudaFieldLocation, float const *, int, identity<float>, float, plus<float>);

  template void transform_reduce<float, float, int, identity<float>, plus<float>>(
    QudaFieldLocation, std::vector<float> &, std::vector<float *> const &, int, identity<float>,
    float, plus<float>);   

} // namespace quda
