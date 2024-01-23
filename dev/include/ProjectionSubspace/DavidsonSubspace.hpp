#ifndef DAVIDSONSUBSPACE_H
#define DAVIDSONSUBSPACE_H

#include <LinearAlgebra/Matrix.hpp>
#include <Orthogonalizations/MGS.hpp>
#include <iostream>

namespace ProjectionSubspace {


template<class fp>
class DavidsonSubspace {

private:
    INT dim;          // dimension of vectors
    INT blockSize;    // number of ritz values active for extraction
    INT maxBasisSize; // number of max block vectors in basis before restart
    INT basisSize;    // current basis size

    std::shared_ptr<LinearAlgebra::Matrix<fp>> V;

    Orthogonalization::MGS<fp> mgsOrth;

public:
    DavidsonSubspace(INT dim_, INT blockSize_, INT maxBasisSize_);

    std::shared_ptr<LinearAlgebra::Matrix<fp>> getRawBasisMatrix();
    INT getRawBasisSize();
    void updateBasis(std::shared_ptr<LinearAlgebra::Matrix<fp>> w);
    INT getMaxBasisSize();
    INT getBlockBasisSize();

};


} // namespace ProjectionSubspace

#include <ProjectionSubspace/DavidsonSubspace.tcc>
#endif // DAVIDSONSUBSPACE_H
