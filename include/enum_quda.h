#pragma once

#include <limits.h>
#define QUDA_INVALID_ENUM INT_MIN

#ifdef __cplusplus
extern "C" {
#endif

typedef enum qudaError_t { QUDA_SUCCESS = 0, QUDA_ERROR = 1, QUDA_ERROR_UNINITIALIZED = 2 } qudaError_t;

typedef enum QudaMemoryType_s {
  QUDA_MEMORY_DEVICE,
  QUDA_MEMORY_PINNED,
  QUDA_MEMORY_MAPPED,
  QUDA_MEMORY_INVALID = QUDA_INVALID_ENUM
} QudaMemoryType;

//
// Types used in QudaGaugeParam
//

typedef enum QudaPrecision_s {
  QUDA_QUARTER_PRECISION = 1,
  QUDA_HALF_PRECISION = 2,
  QUDA_SINGLE_PRECISION = 4,
  QUDA_DOUBLE_PRECISION = 8,
  QUDA_INVALID_PRECISION = QUDA_INVALID_ENUM
} QudaPrecision;


typedef enum QudaVerbosity_s {
  QUDA_SILENT,
  QUDA_SUMMARIZE,
  QUDA_VERBOSE,
  QUDA_DEBUG_VERBOSE,
  QUDA_INVALID_VERBOSITY = QUDA_INVALID_ENUM
} QudaVerbosity;

typedef enum QudaTune_s { QUDA_TUNE_NO, QUDA_TUNE_YES, QUDA_TUNE_INVALID = QUDA_INVALID_ENUM } QudaTune;

typedef enum QudaFieldLocation_s {
  QUDA_CPU_FIELD_LOCATION = 1,
  QUDA_CUDA_FIELD_LOCATION = 2,
  QUDA_INVALID_FIELD_LOCATION = QUDA_INVALID_ENUM
} QudaFieldLocation;


//
// Type used for "parity" argument to dslashQuda()
//

//
// Types used only internally
//

typedef enum QudaBLASOperation_s {
  QUDA_BLAS_OP_N = 0, // No transpose
  QUDA_BLAS_OP_T = 1, // Transpose only
  QUDA_BLAS_OP_C = 2, // Conjugate transpose
  QUDA_BLAS_OP_INVALID = QUDA_INVALID_ENUM
} QudaBLASOperation;

typedef enum QudaBLASDataType_s {
  QUDA_BLAS_DATATYPE_S = 0, // Single
  QUDA_BLAS_DATATYPE_D = 1, // Double
  QUDA_BLAS_DATATYPE_C = 2, // Complex(single)
  QUDA_BLAS_DATATYPE_Z = 3, // Complex(double)
  QUDA_BLAS_DATATYPE_INVALID = QUDA_INVALID_ENUM
} QudaBLASDataType;

typedef enum QudaBLASDataOrder_s {
  QUDA_BLAS_DATAORDER_ROW = 0,
  QUDA_BLAS_DATAORDER_COL = 1,
  QUDA_BLAS_DATAORDER_INVALID = QUDA_INVALID_ENUM
} QudaBLASDataOrder;


typedef enum QudaBoolean_s {
  QUDA_BOOLEAN_FALSE = 0,
  QUDA_BOOLEAN_TRUE = 1,
  QUDA_BOOLEAN_INVALID = QUDA_INVALID_ENUM
} QudaBoolean;

// define these for backwards compatibility
#define QUDA_BOOLEAN_NO QUDA_BOOLEAN_FALSE
#define QUDA_BOOLEAN_YES QUDA_BOOLEAN_TRUE


#ifdef __cplusplus
}
#endif

