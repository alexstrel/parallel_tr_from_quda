#pragma once

#include <quda_define.h>
#include <quda_api.h>

#if defined(QUDA_TARGET_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <string>
#include <complex>
#include <vector>

// this is a helper macro for stripping the path information from
// __FILE__.  FIXME - convert this into a consexpr routine
#define KERNEL_FILE                                                                                                    \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 :                                                               \
                            strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#define TEX_ALIGN_REQ (512*2) //Fermi, factor 2 comes from even/odd
#define ALIGNMENT_ADJUST(n) ( (n+TEX_ALIGN_REQ-1)/TEX_ALIGN_REQ*TEX_ALIGN_REQ)
//#include <quda.h>
#include <util_quda.h>
#include <malloc_quda.h>
#include <object.h>
#include <device.h>

namespace quda {

  using Complex = std::complex<double>;

  /**
   * Check that the resident gauge field is compatible with the requested inv_param
   * @param inv_param   Contains all metadata regarding host and device storage
   */
  class TimeProfile;

}

