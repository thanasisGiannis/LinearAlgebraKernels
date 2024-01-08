#ifndef CHOLESKY_HPP
#define CHOLESKY_HPP

#include <LinearAlgebra/Matrix.hpp>
#include <Orthogonalizations/OrthogonalizationCommon.hpp>
#include <iostream>


namespace Orthogonalization {


template<class fp>
class Cholesky {

private:
    INT rowMaxDim;
    INT colMaxDim;

public:
    Cholesky(const INT dim, const INT nrhs);
    OrthogonalizationErr_t QR(const INT m, const INT n,
                              std::shared_ptr<LinearAlgebra::Matrix<fp>> Q,
                              std::shared_ptr<LinearAlgebra::Matrix<fp>> R);

};


} // namespace Orthogonalization

#include <Orthogonalizations/Cholesky.tcc>
#endif // CHOLESKY_HPP
