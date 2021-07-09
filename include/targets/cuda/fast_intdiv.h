#pragma once

#ifdef _NVHPC_CUDA
#include <nv/target>
#endif

// declaration of class we wish to specialize
template <bool> struct mul_hi;

template <> struct mul_hi<true> {
  __forceinline__ int operator()(const int n, const int m)
  {
    int q;
#ifdef _NVHPC_CUDA    
    if target (nv::target::is_device) {
      asm("mul.hi.s32 %0, %1, %2;" : "=r"(q) : "r"(m), "r"(n));
    } else {
      printf("Instruction is not implemented for x86 marchs.\n");
    }
#else
    printf("Instruction is not implemented for non-nvcpp compiler.\n");
#endif    
    return q;
  }
};

#include "../generic/fast_intdiv.h"
