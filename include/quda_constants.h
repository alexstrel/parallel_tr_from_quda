/**
 * @def   QUDA_MAX_DIM
 * @brief Maximum number of dimensions supported by QUDA.  In practice, no
 *        routines make use of more than 5.
 */
#define QUDA_MAX_DIM 6

/**
 * @def   QUDA_MAX_GEOMETRY
 * @brief Maximum geometry supported by a field.  This essentially is
 * the maximum number of dimensions supported per lattice site.
 */
#define QUDA_MAX_GEOMETRY 8

/**
 * @def QUDA_MAX_MULTI_SHIFT
 * @brief Maximum number of shifts supported by the multi-shift solver.
 *        This number may be changed if need be.
 */
#define QUDA_MAX_MULTI_SHIFT 32

/**
 * @def QUDA_MAX_BLOCK_SRC
 * @brief Maximum number of sources that can be supported by the block solver
 */
#define QUDA_MAX_BLOCK_SRC 64

/**
 * @def QUDA_MAX_ARRAY
 * @brief Maximum array length used in QudaInvertParam arrays
 */
#define QUDA_MAX_ARRAY_SIZE (QUDA_MAX_MULTI_SHIFT > QUDA_MAX_BLOCK_SRC ? QUDA_MAX_MULTI_SHIFT : QUDA_MAX_BLOCK_SRC)

/**
 * @def QUDA_MAX_MULTI_REDUCE
 * @brief Maximum number of simultaneous reductions that can take
 * place.  This number may be increased if needed.
 */
#define QUDA_MAX_MULTI_REDUCE 1024
