#pragma once

//custom reduction operations (DSL programming)

namespace quda {

  template<typename ReduceType, typename Float> struct square_ {
    square_(ReduceType = 1.0) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<Float> &x) const
    { return static_cast<ReduceType>(norm(x)); }
  };

  template <typename ReduceType> struct square_<ReduceType, int8_t> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int8_t> &x) const
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename ReduceType> struct square_<ReduceType,short> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<short> &x) const
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename ReduceType> struct square_<ReduceType,int> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int> &x) const
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename Float, typename storeFloat> struct abs_ {
    abs_(const Float = 1.0) { }
    __host__ __device__ Float operator()(const quda::complex<storeFloat> &x) const
    { return abs(x); }
  };

  template <typename Float> struct abs_<Float, int8_t> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int8_t> &x) const
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

  template<typename Float> struct abs_<Float,short> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<short> &x) const
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

  template<typename Float> struct abs_<Float,int> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int> &x) const
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

}
