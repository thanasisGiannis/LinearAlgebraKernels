#ifndef MGS_HPP
#define MGS_HPP

#include <LinearAlgebra/Matrix.hpp>
#include <Orthogonalizations/OrthogonalizationCommon.hpp>

namespace Orthogonalization {

template<class fp>
class MGS {

public:
    MGS();

    OrthogonalizationErr_t
    QR(const INT m, const INT n,
       std::shared_ptr<LinearAlgebra::Matrix<fp>> Q,
       std::shared_ptr<LinearAlgebra::Matrix<fp>> R);

    OrthogonalizationErr_t
    orth(const INT m, const INT n,
         std::shared_ptr<LinearAlgebra::Matrix<fp>> Q);

    OrthogonalizationErr_t
    orthAgainst(const INT m,
                const INT nQ,
                std::shared_ptr<LinearAlgebra::Matrix<fp>> Q,
                const INT nW,
                std::shared_ptr<LinearAlgebra::Matrix<fp>> W);

};

}

#include <Orthogonalizations/MGS.tcc>

#endif // MGS_HPP
