# all files for quda -- needs some cleanup
# cmake-format: off

# QUDA_HASH for tunecache
if(NOT GITVERSION)
  set(GITVERSION ${PROJECT_VERSION})
endif()
set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${QUDA_GPU_ARCH},cuda_version=${CMAKE_CUDA_COMPILER_VERSION})

# this allows simplified running of clang-tidy
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# build up git version add -debug to GITVERSION if we build with debug options enabled
string(REGEX MATCH [Dd][Ee][Bb][Uu][Gg] DEBUG_BUILD ${CMAKE_BUILD_TYPE})
if(DEBUG_BUILD)
  if(GITVERSION)
    set(GITVERSION ${GITVERSION}-debug)
  else()
    set(GITVERSION debug)
  endif()
endif()

set (QUDA_OBJS
  # cmake-format: sortable
  timer.cpp
  util_quda.cpp
  comm_common.cpp communicator_stack.cpp
  tune.cpp)
# cmake-format: on

# split source into cu and cpp files
foreach(item ${QUDA_OBJS})
  string(REGEX MATCH ".+\\.cu$" item_match ${item})
  if(item_match)
    list(APPEND QUDA_CU_OBJS ${item})
  endif(item_match)
endforeach(item ${QUDA_OBJS})

#list(REMOVE_ITEM QUDA_OBJS ${QUDA_CU_OBJS})

# QUDA_CU_OBJS should contain all cuda files now QUDA_OBJS all c, cpp, fortran sources

# if we have a git version make version.cpp depend on git head so that it is rebuild if the git sha changed
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  if(GITVERSION)
    find_path(
      QUDA_GITDIR NAME HEAD
      PATHS ${CMAKE_SOURCE_DIR}/.git/logs
      NO_DEFAULT_PATH)
    include(AddFileDependencies)
    if(QUDA_GITDIR)
      add_file_dependencies(version.cpp ${QUDA_GITDIR}/HEAD)
    endif()
  endif()
  mark_as_advanced(QUDA_GITDIR)
endif()

# generate a cmake object library for all cpp files first
add_library(quda_cpp OBJECT ${QUDA_OBJS})

# add comms and QIO
target_sources(quda_cpp PRIVATE $<IF:$<BOOL:${QUDA_MPI}>,communicator_mpi.cpp,$<IF:$<BOOL:${QUDA_QMP}>,communicator_qmp.cpp,communicator_single.cpp>>)

# add some deifnitions that cause issues with cmake 3.7 and nvcc only to cpp files
target_compile_definitions(quda_cpp PUBLIC -DQUDA_HASH="${HASH}")
if(GITVERSION)
  target_compile_definitions(quda_cpp PUBLIC -DGITVERSION="${GITVERSION}")
endif()

# make one library
if(QUDA_BUILD_SHAREDLIB)
  set_target_properties(quda_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  add_library(quda SHARED $<TARGET_OBJECTS:quda_cpp> $<$<TARGET_EXISTS:quda_pack>:$<TARGET_OBJECTS:quda_pack>> ${QUDA_CU_OBJS})
  if(CUDAToolkit_FOUND)
    target_link_libraries(quda INTERFACE CUDA::cudart_static)
  endif()
else()
  add_library(quda STATIC $<TARGET_OBJECTS:quda_cpp> $<$<TARGET_EXISTS:quda_pack>:$<TARGET_OBJECTS:quda_pack>> ${QUDA_CU_OBJS})
endif()


# malloc.cpp uses both the driver and runtime api So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub
# version for cmake 3.8 and later this has been integrated into  FindCUDALibs.cmake
target_link_libraries(quda PUBLIC ${CUDA_cuda_driver_LIBRARY})

# set up QUDA compile options
target_compile_definitions(
  quda PRIVATE $<$<CONFIG:DEVEL>:DEVEL> $<$<CONFIG:HOSTDEBUG>:HOST_DEBUG>
               $<$<CONFIG:DEBUG>:HOST_DEBUG> $<$<CONFIG:DEBUG>:DEVICE_DEBUG> $<$<CONFIG:SANITIZE>:HOST_DEBUG>)

target_compile_options(
  quda
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
          -ftz=true
          -prec-div=false
          -prec-sqrt=false>
          $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
          -fcuda-flush-denormals-to-zero
          -fcuda-approx-transcendentals
          -Xclang -fcuda-allow-variadic-functions>)
target_compile_options(
  quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-Wno-unknown-cuda-version> $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
               -Wno-deprecated-gpu-targets -arch=${QUDA_GPU_ARCH} --expt-relaxed-constexpr>)

target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -ftz=true -prec-div=false -prec-sqrt=false>)
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Wno-deprecated-gpu-targets
                                    -arch=${QUDA_GPU_ARCH} --expt-relaxed-constexpr>)
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}
                                    --cuda-gpu-arch=${QUDA_GPU_ARCH}>)
target_link_options(quda PUBLIC $<$<CUDA_COMPILER_ID:Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}>)

if(QUDA_VERBOSE_BUILD)
  target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--ptxas-options=-v>)
endif(QUDA_VERBOSE_BUILD)

if (${CMAKE_CUDA_COMPILER_ID} MATCHES "NVHPC" AND NOT ${CMAKE_BUILD_TYPE} MATCHES "DEBUG")
  target_compile_options(quda PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVHPC>:SHELL: -gpu=nodebug" >)
endif()


# workaround for 10.2
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA"
   AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10.2"
   AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "10.3")
  target_compile_options(
    quda PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL: -Xcicc \"--Xllc -dag-vectorize-ops=1\" " >)
endif()
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:--cuda-path=${CUDAToolkit_TARGET_DIR}>)

target_include_directories(quda PRIVATE .)
target_include_directories(quda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quda SYSTEM PRIVATE ../include/externals)
target_include_directories(quda SYSTEM PUBLIC $<$<OR:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<COMPILE_LANG_AND_ID:CUDA,Clang>>:${CUDAToolkit_INCLUDE_DIRS}>)

target_include_directories(quda SYSTEM PRIVATE ${EIGEN_INCLUDE_DIRS})
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
                                       $<INSTALL_INTERFACE:include/>)
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)

target_include_directories(quda_cpp SYSTEM PRIVATE ../include/externals)
target_include_directories(quda_cpp SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(quda_cpp SYSTEM PRIVATE ${EIGEN_INCLUDE_DIRS})
target_compile_definitions(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,COMPILE_DEFINITIONS>)
target_include_directories(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,INCLUDE_DIRECTORIES>)
target_compile_options(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,COMPILE_OPTIONS>)

# nvshmem enabled parts need CUDA_SEPARABLE_COMPILATION ...
if(QUDA_NVSHMEM)
  add_library(quda_pack OBJECT ${QUDA_DSLASH_OBJS})
  target_include_directories(quda_pack PRIVATE dslash_core)
  target_include_directories(quda_pack SYSTEM PRIVATE ../include/externals)
  target_include_directories(quda_pack PRIVATE .)
  set_target_properties(quda_pack PROPERTIES POSITION_INDEPENDENT_CODE ${QUDA_BUILD_SHAREDLIB})
  target_compile_definitions(quda_pack PRIVATE $<TARGET_PROPERTY:quda,COMPILE_DEFINITIONS>)
  target_include_directories(quda_pack PRIVATE $<TARGET_PROPERTY:quda,INCLUDE_DIRECTORIES>)
  target_compile_options(quda_pack PRIVATE $<TARGET_PROPERTY:quda,COMPILE_OPTIONS>)
  set_target_properties(quda_pack PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET quda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# add target specific files		

if(${QUDA_TARGET_TYPE} STREQUAL "CUDA")		
  add_subdirectory(targets/cuda)		
  target_include_directories(quda PRIVATE ../include/targets/cuda)
  target_include_directories(quda PRIVATE ../include/kernels)  
  target_include_directories(quda SYSTEM PRIVATE ../include/targets/cuda/externals)
endif()		
if(${QUDA_TARGET_TYPE} STREQUAL "HIP")		
  add_subdirectory(targets/hip)
  target_include_directories(quda PRIVATE ../include/targets/hip)
endif()

add_subdirectory(targets/generic)
target_include_directories(quda PRIVATE ../include/targets/generic)

#add_subdirectory(interface)

# propagate CXX flags to CUDA host compiler
#TODO: Do we still need that? 
if(${QUDA_PROPAGATE_CXX_FLAGS})

  # Pick the right set of flags Apparently I cannot do this with generator expressions
  if(CMAKE_BUILD_TYPE STREQUAL "DEVEL")
    set(PROPAGATED_FLAGS "${CMAKE_CXX_FLAGS_DEVEL}")
  elseif(CMAKE_BUILD_TYPE STREQUAL "STRICT")
    set(PROPAGATED_FLAGS "${CMAKE_CXX_FLAGS_STRICT}")
  elseif(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    set(PROPAGATED_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
  elseif(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    set(PROPAGATED_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
  elseif(CMAKE_BUILD_TYPE STREQUAL "HOSTDEBUG")
    set(PROPAGATED_FLAGS "${CMAKE_CXX_FLAGS_HOSTDEBUG}")
  elseif(CMAKE_BUILD_TYPE STREQUAL "SANITIZE")
    set(PROPAGATED_FLAGS "${CMAKE_CXX_FLAGS_SANITIZE}")
  endif()

  # Turne the flags into a CMAKE list
  string(REPLACE " " ";" PROPAGATED_FLAG_LIST ${PROPAGATED_FLAGS})

  foreach(FLAG ${PROPAGATED_FLAG_LIST})
    target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Xcompiler=${FLAG}>)
  endforeach()
endif()

# Specific config dependent warning suppressions and lineinfo forwarding
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
                            -Wreorder
                            $<$<CXX_COMPILER_ID:Clang>: -Xcompiler=-Wno-unused-function -Xcompiler=-Wno-unknown-pragmas>
                            $<$<CXX_COMPILER_ID:GNU>: -Xcompiler=-Wno-unknown-pragmas>
                            $<$<CONFIG:DEVEL>:-Xptxas -warn-lmem-usage,-warn-spills -lineinfo>
                            $<$<CONFIG:STRICT>: -Werror=all-warnings -lineinfo>
                            $<$<CONFIG:HOSTDEBUG>:-lineinfo>
                            $<$<CONFIG:SANITIZE>:-lineinfo>
                            >)

target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
                            -Wall -Wextra -Wno-unknown-pragmas
                            $<$<CONFIG:STRICT>:-Werror -Wno-error=pass-failed>
                            $<$<CONFIG:SANITIZE>:-fsanitize=address -fsanitize=undefined>
                            >)

# some clang warnings should be warning even when turning warnings into errors
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(quda_cpp PUBLIC $<$<COMPILE_LANGUAGE:CXX>: -Wno-error=unused-function>)

  # this is a hack to get colored diagnostics back when using Ninja and clang
  if(CMAKE_GENERATOR MATCHES "Ninja")
    target_compile_options(quda PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
  endif()
endif()

# QUDA FEATURES

# MULTI GPU AND USQCD
if(QUDA_MPI OR QUDA_QMP)
  target_compile_definitions(quda PUBLIC MULTI_GPU)
endif()

if(QUDA_MPI)
  target_link_libraries(quda PUBLIC MPI::MPI_CXX)
  target_compile_definitions(quda PUBLIC MPI_COMMS)
endif()

if(QUDA_NVSHMEM)
  target_link_libraries(quda PUBLIC MPI::MPI_C)
  target_compile_definitions(quda PUBLIC NVSHMEM_COMMS)
  if(QUDA_DOWNLOAD_NVSHMEM)
    add_dependencies(quda NVSHMEM)
    add_dependencies(quda_cpp NVSHMEM)
    add_dependencies(quda_pack NVSHMEM)
  endif()
  get_filename_component(NVSHMEM_LIBPATH ${NVSHMEM_LIBS} DIRECTORY)
  target_link_libraries(quda PUBLIC -L${NVSHMEM_LIBPATH} -lnvshmem)
  target_include_directories(quda SYSTEM PUBLIC $<BUILD_INTERFACE:${NVSHMEM_INCLUDE}>)
endif()


# set which precisions to enable
target_compile_definitions(quda PUBLIC QUDA_PRECISION=${QUDA_PRECISION})
target_compile_definitions(quda PUBLIC QUDA_RECONSTRUCT=${QUDA_RECONSTRUCT})

if(QUDA_FAST_COMPILE_REDUCE)
  target_compile_definitions(quda PRIVATE QUDA_FAST_COMPILE_REDUCE)
endif()

if(QUDA_FAST_COMPILE_DSLASH)
  target_compile_definitions(quda PRIVATE QUDA_FAST_COMPILE_DSLASH)
endif()

if(QUDA_JITIFY)
  target_compile_definitions(quda PRIVATE JITIFY)
  find_package(LibDL)
  target_link_libraries(quda PUBLIC ${CUDA_nvrtc_LIBRARY})
  target_link_libraries(quda PUBLIC ${LIBDL_LIBRARIES})
  target_include_directories(quda PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
endif()

if(QUDA_MPI_NVTX)
  target_sources(quda_cpp PRIVATE nvtx_pmpi.c)
  set(QUDA_NVTX ON)
endif(QUDA_MPI_NVTX)

if(QUDA_INTERFACE_NVTX)
  target_compile_definitions(quda PRIVATE INTERFACE_NVTX)
  set(QUDA_NVTX ON)
endif(QUDA_INTERFACE_NVTX)

if(QUDA_NVTX)
  find_path(NVTX3 "nvtx3/nvToolsExt.h" PATHS ${CUDA_TOOLKIT_INCLUDE} NO_DEFAULT_PATH)
  if(NVTX3)
    target_compile_definitions(quda PRIVATE QUDA_NVTX_VERSION=3)
  else()
    target_link_libraries(quda PUBLIC ${CUDA_nvToolsExt_LIBRARY})
  endif(NVTX3)
endif(QUDA_NVTX)

if(QUDA_BACKWARDS)
  target_include_directories(quda_cpp SYSTEM PRIVATE ${backward-cpp_SOURCE_DIR})
  set_property(
    SOURCE comm_common.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE comm_common.cpp APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  target_link_libraries(quda PUBLIC ${BACKWARD_LIBRARIES})
endif()

if(QUDA_NUMA_NVML)
  target_compile_definitions(quda PRIVATE NUMA_NVML)
  target_sources(quda_cpp PRIVATE numa_affinity.cpp)
  find_package(NVML REQUIRED)
  target_include_directories(quda PRIVATE SYSTEM NVML_INCLUDE_DIR)
  target_link_libraries(quda PUBLIC ${NVML_LIBRARY})
endif(QUDA_NUMA_NVML)

if(QUDA_NVML)
  target_link_libraries(quda PUBLIC ${NVML_LIBRARY})
endif()

configure_file(../include/quda_define.h.in ../include/quda_define.h @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/include/quda_define.h" DESTINATION include/)

# until we define an install step copy the include directory to the build directory
add_custom_command(
  TARGET quda
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)

# some hackery to prevent having old shared / static builds of quda messing with the current build
add_custom_command(
  TARGET quda
  PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libquda.a ${CMAKE_CURRENT_BINARY_DIR}/libquda.so)

install(
  TARGETS quda
  EXPORT qudaTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES
  DESTINATION include)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/qudaConfigVersion.cmake"
  VERSION ${QUDA_VERSION}
  COMPATIBILITY AnyNewerVersion)

export(
  EXPORT qudaTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/qudaTargets.cmake"
  NAMESPACE quda::)

set(ConfigPackageLocation lib/cmake/quda/)

install(
  EXPORT qudaTargets
  NAMESPACE quda::
  DESTINATION ${ConfigPackageLocation})

add_custom_target(
  mpi_nvtx
  ${PYTHON_EXECUTABLE} generate/wrap.py -g -o nvtx_pmpi.c generate/nvtx.w
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating mpi_nvtx wrapper")
