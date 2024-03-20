#ifndef DAVIDSONSUBSPACE_H
#define DAVIDSONSUBSPACE_H

#include <LinearAlgebra/Matrix.hpp>
#include <Orthogonalizations/MGS.hpp>
#include <iostream>

namespace Subspace {

template<class fp>
class SubspaceBasis
: public LinearAlgebra::Matrix<fp> {
private:
    INT blockSize;    // number of ritz values active for extraction
    INT maxBasisSize; // number of max block vectors in basis before restart
    INT basisSize;    // current basis size
public:
    SubspaceBasis(INT dim_, INT blockSize_, INT maxBasisSize_);
    INT getBasisSize();
    INT getMaxBasisSize();
    INT getBlockBasisSize();
    INT Rows();
    INT ld();
    INT Cols() override;
    INT size() override;
    auto operator[](INT index);
    void clear();
    void insertDirection(std::shared_ptr<LinearAlgebra::Matrix<fp>> w);

};

template<class fp>
class SubspaceProjection
: public LinearAlgebra::Matrix<fp> {
private:
    INT blockSize;    // number of ritz values active for extraction
    INT maxBasisSize; // number of max block vectors in basis before restart
    INT basisSize;    // current basis size
public:
    SubspaceProjection(INT blockSize_, INT maxBasisSize_);
    INT getBasisSize();
    INT getMaxBasisSize();
    INT getBlockBasisSize();
    INT Rows() override;
    INT ld();
    INT Cols() override;
    INT size() override;
    auto operator[](INT index);
    void clear();

    void updateProjection(
         std::shared_ptr<Subspace::SubspaceBasis<fp>> V,
         std::shared_ptr<LinearAlgebra::Matrix<fp>> Aw);
};

template<class fp>
class SubspaceHandler {

private:
    INT dim;          // dimension of vectors
    INT blockSize;    // number of ritz values active for extraction
    INT maxBasisSize; // number of max block vectors in basis before restart
    INT basisSize;    // current basis size

    Orthogonalization::MGS<fp> mgsOrth;
public:
    SubspaceHandler(std::shared_ptr<SubspaceBasis<fp>> V_);
    void updateBasis(std::shared_ptr<Subspace::
                                        SubspaceBasis<fp>> V,
                     std::shared_ptr<LinearAlgebra::Matrix<fp>> w);

    void orthDirection(std::shared_ptr<Subspace::
                                        SubspaceBasis<fp>> V,
                     std::shared_ptr<LinearAlgebra::Matrix<fp>> w);

    void updateProjection(std::shared_ptr<LinearAlgebra::Matrix<fp>> Aw,
                          std::shared_ptr<Subspace
                            ::SubspaceBasis<fp>> V,
                          std::shared_ptr<Subspace
                            ::SubspaceProjection<fp>> H);

    void restartBasis(std::shared_ptr<Subspace::SubspaceBasis<fp>> V,
                      std::shared_ptr<LinearAlgebra::Matrix<fp>> Xprev,
                      std::shared_ptr<LinearAlgebra::Matrix<fp>> X,
                      std::shared_ptr<LinearAlgebra::Matrix<fp>> w);

};

} // namespace Subspace
#include <ProjectionSubspace/Subspace.tcc>
#endif // DAVIDSONSUBSPACE_H
