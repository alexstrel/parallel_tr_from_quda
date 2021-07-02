#include <transform_reduce_impl.h>
//implementations:
#include <reducer.h>
#include <transformer.h>

namespace quda
{
  // explicit instantiation list for transform_reduce
  template float transform_reduce<QudaFieldLocation, float, int, plus<float>, identity<float>>(
    QudaFieldLocation&, int, int, float, plus<float>, identity<float>);

} // namespace quda
