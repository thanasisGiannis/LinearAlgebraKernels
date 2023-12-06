#ifndef LINEARALGEBRA_OPERATIONS_HPP
#define LINEARALGEBRA_OPERATIONS_HPP

#include <blas.hh>
//#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>

#include <thrust/host_vector.h>

#if defined( BLAS_HAVE_CUBLAS ) \
    || defined( BLAS_HAVE_ROCBLAS ) \
    || defined( BLAS_HAVE_SYCL )

#include <thrust/device_vector.h>
#endif


namespace {
typedef unsigned int UINT;
typedef int64_t INT; // used by mkl library-strongly connected to system arch
}

namespace LinearAlgebra {

// in order to hide the thrust
// and blas namespaces
// we use LinearAlgebra namespace
using namespace thrust;

// defines are from device.hh of BLASPP library
#if defined( BLAS_HAVE_CUBLAS ) \
    || defined( BLAS_HAVE_ROCBLAS ) \
    || defined( BLAS_HAVE_SYCL )

template <class fp>
using vector = thrust::device_vector<fp>;

static blas::Queue blasQueue(0);

// =============================================================================
// Level 1 BLAS template implementations
#define asum(...)  asum(__VA_ARGS__, LinearAlgebra::blasQueue )
#define axpy(...)  axpy(__VA_ARGS__, LinearAlgebra::blasQueue )
//#define copy(...)  copy(__VA_ARGS__, LinearAlgebra::blasQueue )
//#define dot(...)   dot(__VA_ARGS__, LinearAlgebra::blasQueue )
template<class fp>
fp dot(INT n, fp const *x, INT incx, fp const *y, INT incy)
{
    fp result;
    blas::dot(n, x, incx, y, incy, &result, LinearAlgebra::blasQueue);
    return result;
}

#define dotu(...)  dotu(__VA_ARGS__, LinearAlgebra::blasQueue )
//#define iamax(...) iamax(__VA_ARGS__, LinearAlgebra::blasQueue )
//#define nrm2(...)  nrm2(__VA_ARGS__, LinearAlgebra::blasQueue )
template<class fp>
fp nrm2 (INT n, fp const *x, INT incx)
{
    fp result;
    blas::nrm2(n, x, incx, &result, LinearAlgebra::blasQueue);
    return result;
}

#define rot(...)   rot(__VA_ARGS__, LinearAlgebra::blasQueue )
#define rotg(...)  rotg(__VA_ARGS__, LinearAlgebra::blasQueue )
#define rotm(...)  rotm(__VA_ARGS__, LinearAlgebra::blasQueue )
#define rotmg(...) rotmg(__VA_ARGS__, LinearAlgebra::blasQueue )
#define scal(...)  scal(__VA_ARGS__, LinearAlgebra::blasQueue )
//#define swap(...)  swap(__VA_ARGS__, LinearAlgebra::blasQueue )

// =============================================================================
// Level 2 BLAS template implementations

#define gemv(...)  gemv(__VA_ARGS__, LinearAlgebra::blasQueue )
#define ger(...)   ger(__VA_ARGS__, LinearAlgebra::blasQueue )
#define geru(...)  geru(__VA_ARGS__, LinearAlgebra::blasQueue )
#define hemv(...)  hemv(__VA_ARGS__, LinearAlgebra::blasQueue )
#define her(...)   her(__VA_ARGS__, LinearAlgebra::blasQueue )
#define her2(...)  her2(__VA_ARGS__, LinearAlgebra::blasQueue )
#define symv(...)  symv(__VA_ARGS__, LinearAlgebra::blasQueue )
#define syr(...)   syr(__VA_ARGS__, LinearAlgebra::blasQueue )
#define syr2(...)  syr2(__VA_ARGS__, LinearAlgebra::blasQueue )
#define trmv(...)  trmv(__VA_ARGS__, LinearAlgebra::blasQueue )
#define trsv(...)  trsv(__VA_ARGS__, LinearAlgebra::blasQueue )

// =============================================================================
// Level 3 BLAS template implementations

#define gemm(...)  gemm(__VA_ARGS__, LinearAlgebra::blasQueue )
#define hemm(...)  hemm(__VA_ARGS__, LinearAlgebra::blasQueue )
#define herk(...)  herk(__VA_ARGS__, LinearAlgebra::blasQueue )
#define her2k(...) her2k(__VA_ARGS__, LinearAlgebra::blasQueue )
#define symm(...)  symm(__VA_ARGS__, LinearAlgebra::blasQueue )
#define syrk(...)  syrk(__VA_ARGS__, LinearAlgebra::blasQueue )
#define syr2k(...) syr2k(__VA_ARGS__, LinearAlgebra::blasQueue )
#define trmm(...)  trmm(__VA_ARGS__, LinearAlgebra::blasQueue )
#define trsm(...)  trsm(__VA_ARGS__, LinearAlgebra::blasQueue )

namespace Operation=blas;

#else
// ToDo: choose between host and device vectors
template <class fp>
using vector = thrust::host_vector<fp>;
namespace Operation=blas;
#endif

}
#endif // LINEARALGEBRA_OPERATIONS_HPP
