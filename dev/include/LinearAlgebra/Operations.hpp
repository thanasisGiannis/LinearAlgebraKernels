#ifndef LINEARALGEBRA_OPERATIONS_HPP
#define LINEARALGEBRA_OPERATIONS_HPP


#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/execution_policy.h>

#include <blas.hh>

#include <thrust/host_vector.h>

#if defined(DEVICE)
#include "Operations_Device.hpp"
#else
#include <lapack.hh>
#endif

namespace {
    typedef unsigned int UINT;
    typedef int64_t INT; // used by mkl library-strongly connected to system arch
}



namespace LinearAlgebra {
    using namespace thrust;

#if defined(DEVICE)
    // our hidden data structure of vectors is chosen here
    template <class fp>
    using vector = thrust::device_vector<fp>;

    #include "Operations_Device.hpp"
#else
    // in order to hide the thrust
    // and blas namespaces
    // we use LinearAlgebra namespace
    namespace Operation
    {
        using namespace blas;
        using namespace lapack;
    }

    // just use host_vector and every call to blaspp will be forwared automatically
    // as we use the same interface 
    template <class fp>
    using vector = thrust::host_vector<fp>;
#endif
}

#endif // LINEARALGEBRA_OPERATIONS_HPP
