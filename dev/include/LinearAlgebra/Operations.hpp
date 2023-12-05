#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include <blas.hh>
#if defined( BLAS_HAVE_CUBLAS ) \
    || defined( BLAS_HAVE_ROCBLAS ) \
    || defined( BLAS_HAVE_SYCL )

#include <thrust/device_vector.h>
#else
#include <thrust/host_vector.h>
#endif

#include <thrust/copy.h>
#include <thrust/fill.h>

namespace {
typedef unsigned int UINT;
typedef int64_t INT; // used by mkl library-strongly connected to system arch
}

namespace LinearAlgebra {

// in order to hide the thrust
// and blas namespaces
// we use LinearAlgebra namespace
// e.g. instead of blas::gemm()
// use LinearAlgebra::gemm()
using namespace thrust;
namespace Operation=blas;

// defines are from device.hh of BLASPP library
#if defined( BLAS_HAVE_CUBLAS ) \
    || defined( BLAS_HAVE_ROCBLAS ) \
    || defined( BLAS_HAVE_SYCL )

template <class fp>
using vector = thrust::device_vector<fp>;

static std::shared_ptr<blas::Queue> queue = std::make_shared(new blas::Queue());

blas::Queue Q = queue.get();

// =============================================================================
// Level 1 BLAS template implementations
#define asum(...)  asum(__VA_ARGS__, Q )
#define axpy(...)  axpy(__VA_ARGS__, Q)
#define copy(...)  copy(__VA_ARGS__, Q)
#define dot(...)   dot(__VA_ARGS__, Q)
#define dotu(...)  dotu(__VA_ARGS__, Q)
#define iamax(...) iamax(__VA_ARGS__, Q)
#define nrm2(...)  nrm2(__VA_ARGS__, Q)
#define rot(...)   rot(__VA_ARGS__, Q)
#define rotg(...)  rotg(__VA_ARGS__, Q)
#define rotm(...)  rotm(__VA_ARGS__, Q)
#define rotmg(...) rotmg(__VA_ARGS__, Q)
#define scal(...)  scal(__VA_ARGS__, Q)
#define swap(...)  swap(__VA_ARGS__, Q)

// =============================================================================
// Level 2 BLAS template implementations

#define gemv(...)  gemv(__VA_ARGS__, Q)
#define ger(...)   ger(__VA_ARGS__, Q)
#define geru(...)  geru(__VA_ARGS__, Q)
#define hemv(...)  hemv(__VA_ARGS__, Q)
#define her(...)   her(__VA_ARGS__, Q)
#define her2(...)  her2(__VA_ARGS__, Q)
#define symv(...)  symv(__VA_ARGS__, Q)
#define syr(...)   syr(__VA_ARGS__, Q)
#define syr2(...)  syr2(__VA_ARGS__, Q)
#define trmv(...)  trmv(__VA_ARGS__, Q)
#define trsv(...)  trsv(__VA_ARGS__, Q)

// =============================================================================
// Level 3 BLAS template implementations

#define gemm(...)  gemm(__VA_ARGS__, Q)
#define hemm(...)  hemm(__VA_ARGS__, Q)
#define herk(...)  herk(__VA_ARGS__, Q)
#define her2k(...) her2k(__VA_ARGS__, Q)
#define symm(...)  symm(__VA_ARGS__, Q)
#define syrk(...)  syrk(__VA_ARGS__, Q)
#define syr2k(...) syr2k(__VA_ARGS__, Q)
#define trmm(...)  trmm(__VA_ARGS__, Q)
#define trsm(...)  trsm(__VA_ARGS__, Q)

#else
// ToDo: choose between host and device vectors
template <class fp>
using vector = thrust::host_vector<fp>;
#endif

}
#endif // OPERATIONS_HPP
