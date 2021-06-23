# basic setup for cmake
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# disable gnu exentions
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

# disable in source builds this is only a temporary fix, but for now we need it as cmake will otherwise overwrite the
# existing makefiles
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
# add a directory for cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Git)
find_package(PythonInterp)

# by default we will build DEVEL
if(DEFINED ENV{QUDA_BUILD_TYPE})
  set(DEFBUILD $ENV{QUDA_BUILD_TYPE})
else()
  set(DEFBUILD "DEVEL")
endif()

set(VALID_BUILD_TYPES
    DEVEL
    RELEASE
    STRICT
    DEBUG
    HOSTDEBUG
    SANITIZE)
set(CMAKE_BUILD_TYPE
    "${DEFBUILD}"
    CACHE STRING "Choose the type of build, options are: ${VALID_BUILD_TYPES}")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${VALID_BUILD_TYPES})

string(TOUPPER ${CMAKE_BUILD_TYPE} CHECK_BUILD_TYPE)
list(FIND VALID_BUILD_TYPES ${CHECK_BUILD_TYPE} BUILD_TYPE_VALID)

if(BUILD_TYPE_VALID LESS 0)
  message(SEND_ERROR "Please specify a valid CMAKE_BUILD_TYPE type! Valid build types are:" "${VALID_BUILD_TYPES}")
endif()

# Target type
if(DEFINED ENV{QUDA_TARGET})
  set(DEFTARGET $ENV{QUDA_TARGET})
else()
  set(DEFTARGET "CUDA")
endif()

set(VALID_TARGET_TYPES CUDA HIP)
set(QUDA_TARGET_TYPE
    "${DEFTARGET}"
    CACHE STRING "Choose the type of target, options are: ${VALID_TARGET_TYPES}")
set_property(CACHE QUDA_TARGET_TYPE PROPERTY STRINGS CUDA HIP)

string(TOUPPER ${QUDA_TARGET_TYPE} CHECK_TARGET_TYPE)
list(FIND VALID_TARGET_TYPES ${CHECK_TARGET_TYPE} TARGET_TYPE_VALID)

if(TARGET_TYPE_VALID LESS 0)
  message(SEND_ERROR "Please specify a valid QUDA_TARGET_TYPE type! Valid target types are:" "${VALID_TARGET_TYPES}")
endif()

if( ${CHECK_TARGET_TYPE} STREQUAL "CUDA")
  set(QUDA_TARGET_CUDA ON)
  set(QUDA_TARGET_LIBRARY quda_cuda_target)
endif()

#
# PROJECT is QUDA
#
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} show
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE IS_GIT_REPOSIITORY
    OUTPUT_QUIET ERROR_QUIET)
  if(${IS_GIT_REPOSIITORY} EQUAL 0)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --abbrev=0
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GITTAG OUTPUT_STRIP_TRAILING_WHITESPACE)
    # we use git rev-list and pipe that through wc here. Newer git versions support --count as option to rev-list but
    # that might not always be available
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-list ${GITTAG}..HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMAND wc -l
      OUTPUT_VARIABLE GITCOUNT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --long --dirty
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GITVERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
endif(GIT_FOUND)

project(
  "QUDA"
  VERSION 1.1.0
  LANGUAGES)

message(STATUS "")
message(STATUS "${PROJECT_NAME} ${PROJECT_VERSION} (${GITVERSION}) **")
message(STATUS "cmake version: ${CMAKE_VERSION}")
message(STATUS "Source location: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build location: ${CMAKE_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "QUDA target: ${QUDA_TARGET_TYPE}")

# ######################################################################################################################
# QUDA OPTIONS likely to be changed by users
# ######################################################################################################################
if(DEFINED ENV{QUDA_GPU_ARCH})
  set(QUDA_DEFAULT_GPU_ARCH $ENV{QUDA_GPU_ARCH})
else()
  set(QUDA_DEFAULT_GPU_ARCH sm_70)
endif()
if(NOT QUDA_GPU_ARCH)
  message(STATUS "Building QUDA for GPU ARCH " "${QUDA_DEFAULT_GPU_ARCH}")
endif()

set(QUDA_GPU_ARCH
    ${QUDA_DEFAULT_GPU_ARCH}
    CACHE STRING "set the GPU architecture (sm_60, sm_70, sm_80)")
set_property(CACHE QUDA_GPU_ARCH PROPERTY STRINGS sm_60 sm_70 sm_80)
# build options
option(QUDA_DIRAC_DEFAULT_OFF "default value for QUDA_DIRAC_<TYPE> setting" $ENV{QUDA_DIRAC_DEFAULT_OFF})
mark_as_advanced(QUDA_DIRAC_DEFAULT_OFF)
if(QUDA_DIRAC_DEFAULT_OFF)
  set(QUDA_DIRAC_DEFAULT OFF)
else()
  set(QUDA_DIRAC_DEFAULT ON)
endif()


# Multi-GPU options
option(QUDA_QMP "build the QMP multi-GPU code" OFF)


# Locations for QIO / QMP

# ######################################################################################################################
# QUDA ADVANCED OPTIONS that usually should not be changed by users
# ######################################################################################################################
option(QUDA_BUILD_ALL_TESTS "build tests by default" ON)
option(QUDA_INSTALL_ALL_TESTS "install tests by default" ON)
option(QUDA_BUILD_SHAREDLIB "build quda as a shared lib" ON)
option(QUDA_PROPAGATE_CXX_FLAGS "propagate CXX_FLAGS to CUDA host compiler (for cmake >= 3.8)" ON)
option(QUDA_FLOAT8 "enable float-8 ordered fixed-point fields?" OFF)
option(QUDA_NVML "use NVML to report CUDA graphics driver version" OFF)
option(QUDA_NUMA_NVML "experimental use of NVML to set numa affinity" OFF)
option(QUDA_VERBOSE_BUILD "display kernel register usage" OFF)
option(QUDA_BUILD_NATIVE_LAPACK "build the native blas/lapack library according to QUDA_TARGET" ON)

set(QUDA_MAX_MULTI_BLAS_N
    "4"
    CACHE STRING "maximum value to initialize template for multi-blas /-reduce")
if(QUDA_MAX_MULTI_BLAS_N GREATER 32)
  message(SEND_ERROR "Maximum QUDA_MAX_MULTI_BLAS_N is 32.")
endif()

set(QUDA_PRECISION
    "14"
    CACHE STRING "which precisions to instantiate in QUDA (4-bit number - double, single, half, quarter)")
set(QUDA_RECONSTRUCT
    "7"
    CACHE STRING "which reconstructs to instantiate in QUDA (3-bit number - 18, 13/12, 9/8)")

set(QUDA_NVSHMEM OFF CACHE BOOL "set to 'yes' to build the NVSHMEM multi-GPU code")
set(QUDA_NVSHMEM_HOME $ENV{NVSHMEM_HOME} CACHE PATH "path to NVSHMEM")

# Set CTest options
option(QUDA_CTEST_SEP_DSLASH_POLICIES "Test Dslash policies separately in ctest instead of only autotuning them." OFF)
option(QUDA_CTEST_DISABLE_BENCHMARKS "Disable benchmark test" ON)

option(QUDA_FAST_COMPILE_REDUCE "enable fast compilation in blas and reduction kernels (single warp per reduction)" OFF)
option(QUDA_FAST_COMPILE_DSLASH "enable fast compilation in dslash kernels (~20% perf impact)" OFF)

option(QUDA_OPENMP "enable OpenMP" OFF)
set(QUDA_CXX_STANDARD
    14
    CACHE STRING "set the CXX Standard (14 or 17)")
set_property(CACHE QUDA_CXX_STANDARD PROPERTY STRINGS 14 17)

option(QUDA_BACKWARDS "Enable stacktrace generation using backwards-cpp")


# NVTX options
option(QUDA_MPI_NVTX "add NVTX markup to MPI API calls" OFF)
option(QUDA_INTERFACE_NVTX "add NVTC markup to interface calls" OFF)

# features in development
option(QUDA_DOWNLOAD_NVSHMEM "Download NVSHMEM" OFF)
set(QUDA_DOWNLOAD_NVSHMEM_TAR "" CACHE STRING "location of NVSHMEM tarball")
set(QUDA_GDRCOPY_HOME "/usr/local/gdrcopy" CACHE STRING "path to gdrcopy used when QUDA_DOWNLOAD_NVSHMEM is enabled")


option(QUDA_GENERATE_DOXYGEN "generate doxygen documentation")

# mark as advanced
mark_as_advanced(QUDA_BUILD_ALL_TESTS)
mark_as_advanced(QUDA_INSTALL_ALL_TESTS)
mark_as_advanced(QUDA_PROPAGATE_CXX_FLAGS)
mark_as_advanced(QUDA_HETEROGENEOUS_ATOMIC)
mark_as_advanced(QUDA_FLOAT8)
mark_as_advanced(QUDA_FAST_COMPILE_REDUCE)
mark_as_advanced(QUDA_FAST_COMPILE_DSLASH)
mark_as_advanced(QUDA_NVML)
mark_as_advanced(QUDA_NUMA_NVML)
mark_as_advanced(QUDA_VERBOSE_BUILD)
mark_as_advanced(QUDA_MAX_MULTI_BLAS_N)
mark_as_advanced(QUDA_PRECISION)
mark_as_advanced(QUDA_RECONSTRUCT)
mark_as_advanced(QUDA_CTEST_SEP_DSLASH_POLICIES)
mark_as_advanced(QUDA_CTEST_LAUNCH)
mark_as_advanced(QUDA_CTEST_LAUNCH_ARGS)
mark_as_advanced(QUDA_OPENMP)

mark_as_advanced(QUDA_BACKWARDS)


mark_as_advanced(QUDA_DOWNLOAD_NVSHMEM)
mark_as_advanced(QUDA_DOWNLOAD_NVSHMEM_TAR)
mark_as_advanced(QUDA_GDRCOPY_HOME)
mark_as_advanced(QUDA_MPI_NVTX)
mark_as_advanced(QUDA_INTERFACE_NVTX)
mark_as_advanced(QUDA_INTERFACE_ALL)

mark_as_advanced(QUDA_CXX_STANDARD)

# some checks for invalid combinations

if(QUDA_MPI AND QUDA_QMP)
  message(
    SEND_ERROR
      "Specifying QUDA_QMP and QUDA_MPI might result in undefined behavior. If you intend to use QMP set QUDA_MPI=OFF.")
endif()

if(QUDA_NVSHMEM AND NOT (QUDA_QMP OR QUDA_MPI))
message(
  SEND_ERROR
    "Specifying QUDA_NVSHMEM requires either QUDA_QMP or QUDA_MPI.")
endif()

# COMPILER FLAGS Linux: CMAKE_HOST_SYSTEM_PROCESSOR "x86_64" Mac: CMAKE_HOST_SYSTEM_PROCESSOR "x86_64" Power:
# CMAKE_HOST_SYSTEM_PROCESSOR "ppc64le"

# We need to use different optimization flags depending on whether we are on x86 or power Note: This only applies to the
# RELEASE build type this is just a quick fix and we should probably use
# https://cmake.org/cmake/help/latest/module/CheckCXXCompilerFlag.html

set(CPU_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
if(${CPU_ARCH} STREQUAL "x86_64")
  set(CXX_OPT "-march=native")
elseif(${CPU_ARCH} STREQUAL "ppc64le")
  set(CXX_OPT "-Ofast -mcpu=native -mtune=native")
endif()

set(CMAKE_CXX_STANDARD ${QUDA_CXX_STANDARD})

# define CXX FLAGS
set(CMAKE_CXX_FLAGS_DEVEL
    "-g -O3 -Wall -Wextra"
    CACHE STRING "Flags used by the C++ compiler during regular development builds.")
set(CMAKE_CXX_FLAGS_STRICT
    "-O3 -Werror -Wall -Wextra"
    CACHE STRING "Flags used by the C++ compiler during strict jenkins builds.")
set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -w ${CXX_OPT} "
    CACHE STRING "Flags used by the C++ compiler during release builds.")
set(CMAKE_CXX_FLAGS_HOSTDEBUG
    "-Wall -Wextra -g"
    CACHE STRING "Flags used by the C++ compiler during host-debug builds.")
set(CMAKE_CXX_FLAGS_DEBUG
    "-Wall -Wextra -g -fno-inline"
    CACHE STRING "Flags used by the C++ compiler during full (host+device) debug builds.")
set(CMAKE_CXX_FLAGS_SANITIZE
    "-Wall -Wextra -g -fno-inline \"-fsanitize=address,undefined\""
    CACHE STRING "Flags used by the C++ compiler during santizer debug builds.")

enable_language(CXX)

# define C FLAGS
set(CMAKE_C_FLAGS_DEVEL
    "-g -O3 -Wall -Wextra"
    CACHE STRING "Flags used by the C compiler during regular development builds.")
set(CMAKE_C_FLAGS_STRICT
    "-O3 -Werror -Wall -Wextra"
    CACHE STRING "Flags used by the C compiler during strict jenkins builds.")
set(CMAKE_C_FLAGS_RELEASE
    "-O3 -w"
    CACHE STRING "Flags used by the C compiler during release builds.")
set(CMAKE_C_FLAGS_HOSTDEBUG
    "-Wall -Wextra -g"
    CACHE STRING "Flags used by the C compiler during host-debug builds.")
set(CMAKE_C_FLAGS_DEBUG
    "-Wall -Wextra -g -fno-inline"
    CACHE STRING "Flags used by the C compiler during full (host+device) debug builds.")
set(CMAKE_C_FLAGS_SANITIZE
    "-Wall -Wextra -g -fno-inline \"-fsanitize=address,undefined\""
    CACHE STRING "Flags used by the C compiler during sanitizer debug builds.")

enable_language(C)

# define LINKER FLAGS
set(CMAKE_EXE_LINKER_FLAGS_SANITIZE
    "-fsanitize=address,undefined"
    CACHE STRING "Flags used by the linker during sanitizer debug builds.")

# define CUDA flags
set(CMAKE_CUDA_HOST_COMPILER
    "${CMAKE_CXX_COMPILER}"
    CACHE FILEPATH "Host compiler to be used by nvcc")
set(CMAKE_CUDA_STANDARD ${QUDA_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED True)
mark_as_advanced(CMAKE_CUDA_HOST_COMPILER)

include(CheckLanguage)
check_language(CUDA)

if(${CMAKE_CUDA_COMPILER} MATCHES "nvcc")
  set(QUDA_CUDA_BUILD_TYPE "NVCC")
  message(STATUS "CUDA Build Type: ${QUDA_CUDA_BUILD_TYPE}")
endif()

if(${CMAKE_CUDA_COMPILER} MATCHES "clang")
  if(CMAKE_VERSION VERSION_LESS 3.18)
    message(ERROR "Building QUDA with clang as CMAKE_CUDA_COMPILER requires CMake 3.18+")
  endif()
  set(QUDA_CUDA_BUILD_TYPE "Clang")
  message(STATUS "CUDA Build Type: ${QUDA_CUDA_BUILD_TYPE}")
endif()

set(CMAKE_CUDA_FLAGS_DEVEL
    "-g -O3 "
    CACHE STRING "Flags used by the CUDA compiler during regular development builds.")
set(CMAKE_CUDA_FLAGS_STRICT
    "-g -O3"
    CACHE STRING "Flags used by the CUDA compiler during strict jenkins builds.")
set(CMAKE_CUDA_FLAGS_RELEASE
    "-O3 -w"
    CACHE STRING "Flags used by the CUDA compiler during release builds.")
set(CMAKE_CUDA_FLAGS_HOSTDEBUG
    "-g"
    CACHE STRING "Flags used by the C++ compiler during host-debug builds.")
set(CMAKE_CUDA_FLAGS_DEBUG
    "-g -G"
    CACHE STRING "Flags used by the C++ compiler during full (host+device) debug builds.")
set(CMAKE_CUDA_FLAGS_SANITIZE
    "-g "
    CACHE STRING "Flags used by the C++ compiler during sanitizer debug builds.")

# This is needed now GPU ARCH
set(GITVERSION ${GITVERSION}-${QUDA_GPU_ARCH})
string(REGEX REPLACE sm_ "" COMP_CAP ${QUDA_GPU_ARCH})
set(CMAKE_CUDA_ARCHITECTURES ${COMP_CAP})
set(COMP_CAP "${COMP_CAP}0")

enable_language(CUDA)
message(STATUS "CUDA Compiler is" ${CMAKE_CUDA_COMPILER})
message(STATUS "Compiler ID is " ${CMAKE_CUDA_COMPILER_ID})

# CUDA Wrapper for finding libs etc
if(CMAKE_VERSION VERSION_LESS 3.17)
  find_package(CUDAWrapper)
else()
  # for cmake 3.17+ we rely on
  find_package(CUDAToolkit)
endif()


if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0" AND CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
  set(QUDA_HETEROGENEOUS_ATOMIC_SUPPORT ON)
  message(STATUS "Heterogeneous atomics supported: ${QUDA_HETEROGENEOUS_ATOMIC_SUPPORT}")
endif()
include(CMakeDependentOption)
CMAKE_DEPENDENT_OPTION(QUDA_HETEROGENEOUS_ATOMIC "enable heterogeneous atomic support (CUDA >= 11.0)?" ON "QUDA_HETEROGENEOUS_ATOMIC_SUPPORT" OFF)

if((QUDA_HETEROGENEOUS_ATOMIC OR QUDA_NVSHMEM) AND ${CMAKE_BUILD_TYPE} STREQUAL "SANITIZE")
  message(SEND_ERROR "QUDA_HETEROGENEOUS_ATOMIC=ON AND/OR QUDA_NVSHMEM=ON do not support SANITIZE build)")
endif()
if(QUDA_HETEROGENEOUS_ATOMIC AND QUDA_JITIFY)
  message(SEND_ERROR "QUDA_HETEROGENEOUS_ATOMIC=ON does not support JITIFY)")
endif()

if(QUDA_NVSHMEM AND (${COMP_CAP} LESS "700"))
  message(SEND_ERROR "QUDA_NVSHMEM=ON requires at least QUDA_GPU_ARCH=sm_70")
endif()


if(QUDA_MPI OR QUDA_QMP)
  # if we are using MPI and no MPI_<LANG>_COMPILER was specified on the command line check for MPICXX and MPICC
  # environment variables
  if((NOT MPI_CXX_COMPILER) AND DEFINED ENV{MPICXX})
    set(MPI_CXX_COMPILER $ENV{MPICXX})
    set(mpimessage True)
    message(STATUS "Found environment variable MPICXX. Using it for MPI detection: $ENV{MPICXX}")
  endif()
  if((NOT MPI_C_COMPILER) AND DEFINED ENV{MPICC})
    message(STATUS "Found environment variable MPICC. Using it for MPI detection: $ENV{MPICC}")
    set(MPI_C_COMPILER $ENV{MPICC})
    set(mpimessage True)
  endif()
  # I think we don't use that at all but
  if((NOT MPI_Fortran_COMPILER) AND DEFINED ENV{MPIFORT})
    message(STATUS "Found environment variable MPIFORT. Using it for MPI detection: $ENV{MPIFORT}")
    set(MPI_Fortran_COMPILER $ENV{MPIFORT})
    set(mpimessage True)
  endif()
  if(mpimessage)
    message(
      "Found MPIFORT/MPICC/MPICXX environment variables. If this is not what you want please use -DMPI_<LANG>_COMPILER and consult the cmake FindMPI documentation."
    )
  endif()
  # we need to enable Fortran if we want to detect MPI_Fortran_COMPILER
  if(QUDA_ARPACK OR QUDA_OPENBLAS)
    enable_language(Fortran)
  endif()
  find_package(MPI)
endif()

if(QUDA_NVSHMEM)
  if(QUDA_DOWNLOAD_NVSHMEM)
    ExternalProject_Add(NVSHMEM
                        URL ${QUDA_DOWNLOAD_NVSHMEM_TAR}
                        PREFIX nvshmem
                        CONFIGURE_COMMAND ""
                        BUILD_IN_SOURCE ON
                        BUILD_COMMAND  make NVSHMEM_PREFIX=<INSTALL_DIR> NVSHMEM_MPI_SUPPORT=1 GDRCOPY_HOME=${QUDA_GDRCOPY_HOME} install
                        INSTALL_COMMAND ""
                        LOG_INSTALL ON
                        LOG_BUILD ON
                        LOG_DOWNLOAD ON
                        )
    ExternalProject_Get_Property(NVSHMEM INSTALL_DIR)
    set(QUDA_NVSHMEM_HOME ${INSTALL_DIR} CACHE PATH "path to NVSHMEM" FORCE)
    set(NVSHMEM_LIBS  ${INSTALL_DIR}/lib/libnvshmem.a)
    set(NVSHMEM_INCLUDE ${INSTALL_DIR}/include/)   
  else()
    if("${QUDA_NVSHMEM_HOME}" STREQUAL "")
      message( FATAL_ERROR "QUDA_NVSHMEM_HOME must be defined if QUDA_NVSHMEM is set" )
    endif()
      find_library(NVSHMEM_LIBS NAMES nvshmem PATHS "${QUDA_NVSHMEM_HOME}/lib/"  )
      find_path(NVSHMEM_INCLUDE NAMES nvshmem.h PATHS "${QUDA_NVSHMEM_HOME}/include/" )
  endif()
    
  mark_as_advanced(NVSHMEM_LIBS)
  mark_as_advanced(NVSHMEM_INCLUDE)
  add_library(nvshmem_lib STATIC IMPORTED)
  set_target_properties(nvshmem_lib PROPERTIES IMPORTED_LOCATION ${NVSHMEM_LIBS})
  set_target_properties(nvshmem_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(nvshmem_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  set_target_properties(nvshmem_lib PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES CUDA)
endif()


if(QUDA_OPENMP)
  find_package(OpenMP)
endif()

if(QUDA_MAGMA)
endif(QUDA_MAGMA)

# This selects arpack or parpack for Multi GPU
if(QUDA_ARPACK)
endif(QUDA_ARPACK)

if(QUDA_OPENBLAS)
endif(QUDA_OPENBLAS)


# BACKWARDS
if(QUDA_BACKWARDS)
  include(FetchContent)
  FetchContent_Declare(
    backward-cpp
    GIT_REPOSITORY https://github.com/bombela/backward-cpp.git
    GIT_TAG v1.5
    GIT_SHALLOW ON)
  FetchContent_GetProperties(backward-cpp)
  if(NOT backward-cpp_POPULATED)
    FetchContent_Populate(backward-cpp)
  endif()
  include(${backward-cpp_SOURCE_DIR}/BackwardConfig.cmake)
endif()

# this allows simplified running of clang-tidy
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# make the compiler flags an advanced option for all user defined build types (cmake defined build types are advanced by
# default )
mark_as_advanced(CMAKE_CUDA_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CUDA_FLAGS_STRICT)
mark_as_advanced(CMAKE_CUDA_FLAGS_RELEASE)
mark_as_advanced(CMAKE_CUDA_FLAGS_DEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_SANITIZE)

mark_as_advanced(CMAKE_CXX_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CXX_FLAGS_STRICT)
mark_as_advanced(CMAKE_CXX_FLAGS_RELEASE)
mark_as_advanced(CMAKE_CXX_FLAGS_DEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_SANITIZE)

mark_as_advanced(CMAKE_C_FLAGS_DEVEL)
mark_as_advanced(CMAKE_C_FLAGS_STRICT)
mark_as_advanced(CMAKE_C_FLAGS_RELEASE)
mark_as_advanced(CMAKE_C_FLAGS_DEBUG)
mark_as_advanced(CMAKE_C_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_C_FLAGS_SANITIZE)
mark_as_advanced(CMAKE_F_FLAGS)

mark_as_advanced(CMAKE_EXE_LINKER_FLAGS_SANITIZE)

# enable ctest
include(CTest)

# add tests, utils, reference, and quda library
add_subdirectory(lib)
add_subdirectory(core/cuda)
target_include_directories(quda PRIVATE core/cuda/include)
#add_subdirectory(tests)
#add_subdirectory(doc)
