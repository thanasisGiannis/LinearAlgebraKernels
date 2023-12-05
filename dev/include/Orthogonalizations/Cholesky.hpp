#ifndef CHOLESKY_HPP
#define CHOLESKY_HPP

#include <LinearAlgebra/Matrix.hpp>
#include <Orthogonalizations/OrthogonalizationCommon.hpp>
#include <iostream>

namespace Orthogonalization {


template<class fp>
class Cholesky {
    friend class TestCholesky;

private:
    INT rowMaxDim;
    INT colMaxDim;

    LinearAlgebra::Matrix<fp> B;// this vector will be used
                                // to create upper triangular
                                // cholesky matrix

    bool chol(const int n, std::shared_ptr<LinearAlgebra::Matrix<fp>> L);

public:
    Cholesky(const INT dim, const INT nrhs);
    OrthogonalizationErr_t QR(const INT m, const INT n,
                              std::shared_ptr<LinearAlgebra::Matrix<fp>> Q,
                              std::shared_ptr<LinearAlgebra::Matrix<fp>> R);

};


} // namespace Orthogonalization

#include <Orthogonalizations/Cholesky.tcc>
#endif // CHOLESKY_HPP
