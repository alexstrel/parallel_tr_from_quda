# add target specific files / options
target_sources(quda_cpp PRIVATE transform_reduce.cu)

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
