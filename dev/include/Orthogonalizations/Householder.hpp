#ifndef ORTHOGONALIZATION_HOUSEHOLDER_HPP
#define ORTHOGONALIZATION_HOUSEHOLDER_HPP

#include <LinearAlgebra/Matrix.hpp>
#include <Orthogonalizations/OrthogonalizationCommon.hpp>


namespace Orthogonalization {

template<class fp>
class Householder {
public:
    Householder(INT dim, INT nrhs);
    OrthogonalizationErr_t QR(INT m, INT n, std::shared_ptr<LinearAlgebra::Matrix<fp>> Q_,
            std::shared_ptr<LinearAlgebra::Matrix<fp>> R_);

private:
    INT rowMaxDim;
    INT colMaxDim;

    LinearAlgebra::Matrix<fp> hhQz;     // dim x dim
    LinearAlgebra::Matrix<fp> hhx;      // dim x 1
    LinearAlgebra::Matrix<fp> hhv;      // dim x 1
    LinearAlgebra::Matrix<fp> hhu;      // nrhs x 1
    LinearAlgebra::Matrix<fp> hhvhhvt;  // dim x dim
    LinearAlgebra::Matrix<fp> eye;      // dim x dim
};
} // namespace Orthogonalization
#include <Orthogonalizations/Householder.tcc>
#endif // ORTHOGONALIZATION_HOUSEHOLDER_HPP
