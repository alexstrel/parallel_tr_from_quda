# add target specific files / options
target_sources(quda_cpp PRIVATE quda_api.cpp device.cpp malloc.cpp comm_target.cpp)

if(QUDA_BACKWARDS)
  set_property(SOURCE malloc.cpp DIRECTORY ${CMAKE_SOURCE_DIR}/lib APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE malloc.cpp DIRECTORY ${CMAKE_SOURCE_DIR}/lib APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
  .nvFatBinSegment : { *(.nvFatBinSegment) }
  .nv_fatbin : { *(.nv_fatbin) }
}
]=])
target_link_options(quda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xfatbin=-compress-all>)
