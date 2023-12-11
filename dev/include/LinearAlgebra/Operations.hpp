#ifndef LINEARALGEBRA_OPERATIONS_HPP
#define LINEARALGEBRA_OPERATIONS_HPP

#include <blas.hh>
#include <thrust/host_vector.h>

#if defined(DEVICE)
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#endif


#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/execution_policy.h>

namespace {
typedef unsigned int UINT;
typedef int64_t INT; // used by mkl library-strongly connected to system arch
}

#if defined(DEVICE)
namespace LinearAlgebra {
    static blas::Queue blasQueue(0);
}

namespace blas {

template<class fp>
void copy(INT n, thrust::device_ptr<fp> &&x, INT incx,
thrust::device_ptr<fp> &&y, INT incy)
{
    blas::copy(n, x.get(), incx, y.get(), incy, LinearAlgebra::blasQueue);
}

template<class fp>
fp dot(INT n, thrust::device_ptr<fp>&& x, INT incx,
thrust::device_ptr<fp> &&y, INT incy)
{
    fp result;
    blas::dot(n, x.get(), incx, y.get(), incy, &result, LinearAlgebra::blasQueue);
    return result;
}

template<class fp>
fp nrm2 (INT n, thrust::device_ptr<fp> && x, INT incx)
{
    fp result;
    blas::nrm2(n, x.get(), incx, &result, LinearAlgebra::blasQueue);
    return result;
}

}// namespace blas

#endif


namespace LinearAlgebra {

// in order to hide the thrust
// and blas namespaces
// we use LinearAlgebra namespace
using namespace thrust;
namespace Operation=blas;

// defines are from device.hh of BLASPP library
#if defined(DEVICE)

template <class fp>
using vector = thrust::device_vector<fp>;


template <class fp>
auto&& get_raw_data(fp&& x)
{
    return x;
}

template<class fp>
auto get_raw_data(thrust::device_ptr<fp>&& x)
{
    return x.get();
}

#define GET(arg) LinearAlgebra::get_raw_data(arg)
#define PARENS ()
#define EXPAND(...) EXPAND4(EXPAND4(EXPAND4(EXPAND4(__VA_ARGS__))))
#define EXPAND4(...) EXPAND3(EXPAND3(EXPAND3(EXPAND3(__VA_ARGS__))))
#define EXPAND3(...) EXPAND2(EXPAND2(EXPAND2(EXPAND2(__VA_ARGS__))))
#define EXPAND2(...) EXPAND1(EXPAND1(EXPAND1(EXPAND1(__VA_ARGS__))))
#define EXPAND1(...) __VA_ARGS__

#define FOR_EACH(macro, ...)                                    \
  __VA_OPT__(EXPAND(FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define FOR_EACH_HELPER(macro, a1, ...)                         \
  macro(a1)                                                     \
  __VA_OPT__(,)        \
  __VA_OPT__(FOR_EACH_AGAIN PARENS (macro, __VA_ARGS__))
#define FOR_EACH_AGAIN() FOR_EACH_HELPER

// =============================================================================
// Level 1 BLAS template implementations
#define asum(...)  asum(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define axpy(...)  axpy(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
//#define copy(...)  copy(__VA_ARGS__, LinearAlgebra::blasQueue )
//#define dot(...)   dot(__VA_ARGS__, LinearAlgebra::blasQueue )
#define dotu(...)  dotu(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
//#define iamax(...) iamax(__VA_ARGS__, LinearAlgebra::blasQueue )
#define rot(...)   rot(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define rotg(...)  rotg(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define rotm(...)  rotm(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define rotmg(...) rotmg(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define scal(...)  scal(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
//#define swap(...)  swap(__VA_ARGS__, LinearAlgebra::blasQueue )

// =============================================================================
// Level 2 BLAS template implementations

#define gemv(...)  gemv(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define ger(...)   ger(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define geru(...)  geru(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define hemv(...)  hemv(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define her(...)   her(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define her2(...)  her2(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define symv(...)  symv(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define syr(...)   syr(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define syr2(...)  syr2(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define trmv(...)  trmv(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define trsv(...)  trsv(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )

// =============================================================================
// Level 3 BLAS template implementations

#define gemm(...)  gemm(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define hemm(...)  hemm(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define herk(...)  herk(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define her2k(...) her2k(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define symm(...)  symm(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define syrk(...)  syrk(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define syr2k(...) syr2k(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define trmm(...)  trmm(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )
#define trsm(...)  trsm(FOR_EACH(GET,__VA_ARGS__) , LinearAlgebra::blasQueue )

#else
// ToDo: choose between host and device vectors
template <class fp>
using vector = thrust::host_vector<fp>;
#endif
}

#endif // LINEARALGEBRA_OPERATIONS_HPP
