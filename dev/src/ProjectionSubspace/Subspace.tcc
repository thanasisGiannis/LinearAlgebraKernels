//================================
/*
 * class Subspace::SubspaceBasis
 * */
//================================
template<class fp>
Subspace::SubspaceBasis<fp>::
SubspaceBasis(INT dim_, INT blockSize_, INT maxBasisSize_)
: blockSize{blockSize_}
, maxBasisSize(maxBasisSize_)
, basisSize(0)
, LinearAlgebra::Matrix<fp>(dim_, blockSize_*maxBasisSize_)
{
}

template<class fp>
INT Subspace::SubspaceBasis<fp>::
getBasisSize()
{
    return basisSize;
}

template<class fp>
INT Subspace::SubspaceBasis<fp>::
getMaxBasisSize()
{
    return blockSize*maxBasisSize;
}

template<class fp>
INT Subspace::SubspaceBasis<fp>::
getBlockBasisSize()
{
    return blockSize;
}

template<class fp>
INT Subspace::SubspaceBasis<fp>::
Rows()
{
    return static_cast<LinearAlgebra::Matrix<fp>>(*this).Rows();
}

template<class fp>
INT Subspace::SubspaceBasis<fp>::
ld()
{
    return static_cast<LinearAlgebra::Matrix<fp>>(*this).ld();
}

template<class fp>
INT Subspace::SubspaceBasis<fp>::
Cols()
{
    return basisSize*blockSize;
}

template<class fp>
INT Subspace::SubspaceBasis<fp>::
size()
{
    return Rows()*Cols();
}

template<class fp>
auto Subspace::SubspaceBasis<fp>::
operator[](INT index)
{
    return static_cast<LinearAlgebra::Matrix<fp>>(*this)[index];
}

template<class fp>
void Subspace::SubspaceBasis<fp>::
clear()
{
    basisSize=0;
    LinearAlgebra::fill(this->begin(),
                        this->end(),
                        static_cast<fp>(0.0));
}

template<class fp>
void Subspace::SubspaceBasis<fp>::
insertDirection(std::shared_ptr<LinearAlgebra::Matrix<fp>> w)
{
    if( (w->Rows()==static_cast<LinearAlgebra::Matrix<fp>>(*this).Rows()) &&
        (w->Cols() == blockSize))
    {
        auto dim = static_cast<LinearAlgebra::Matrix<fp>>(*this).Rows();
        auto raw_data = this->data();

        LinearAlgebra::Operation::copy(dim*blockSize,
             w->data()+0+0*(w->ld()), 1,
             raw_data+(0+(this->ld())*blockSize*basisSize), 1);
        basisSize++;
    }
}

//================================
/*
 * class Subspace::SubspaceProjection
 * */
//================================
template<class fp>
Subspace::SubspaceProjection<fp>::
SubspaceProjection(INT blockSize_, INT maxBasisSize_)
: blockSize{blockSize_}
, maxBasisSize(maxBasisSize_)
, basisSize(0)
, LinearAlgebra::Matrix<fp>(blockSize_*maxBasisSize_,
                                blockSize_*maxBasisSize_)
{
}
template<class fp>
INT Subspace::SubspaceProjection<fp>::
getBasisSize()
{
    return basisSize;
}

template<class fp>
INT Subspace::SubspaceProjection<fp>::
getMaxBasisSize()
{
    return blockSize*maxBasisSize;
}

template<class fp>
INT Subspace::SubspaceProjection<fp>::
getBlockBasisSize()
{
    return blockSize;
}

template<class fp>
INT Subspace::SubspaceProjection<fp>::
Rows()
{
    return basisSize*blockSize;
}

template<class fp>
INT Subspace::SubspaceProjection<fp>::
ld()
{
    return static_cast<LinearAlgebra::Matrix<fp>>(*this).ld();
}

template<class fp>
INT Subspace::SubspaceProjection<fp>::
Cols()
{
    return basisSize*blockSize;
}

template<class fp>
INT Subspace::SubspaceProjection<fp>::
size()
{
    return Rows()*Cols();
}

template<class fp>
auto Subspace::SubspaceProjection<fp>::
operator[](INT index)
{
    return static_cast<LinearAlgebra::Matrix<fp>>(*this)[index];
}

template<class fp>
void Subspace::SubspaceProjection<fp>::
clear()
{
    basisSize=0;
    LinearAlgebra::fill(this->begin(),
                        this->end(),
                        static_cast<fp>(0.0));
}

template<class fp>
void Subspace::SubspaceProjection<fp>::
updateProjection(
     std::shared_ptr<Subspace::SubspaceBasis<fp>> V,
     std::shared_ptr<LinearAlgebra::Matrix<fp>> Aw)
{
    if( V->Rows() == Aw->Rows() )
    {
        /*
         * keeping the upper triangular part of H
         * H = [ H      V'Aw;
         *      (V'Aw)' w'Aw;
         *     ] == H(:,basisSize*blockSize:end) = V'*Aw
         */

         if(V->Cols() != 0)
         {
             LinearAlgebra::Operation::gemm(blas::Layout::ColMajor,
                                   blas::Op::Trans,
                                   blas::Op::NoTrans,
                                   V->Cols(), blockSize, V->Rows(),
                                   static_cast<double>(1.0),
                                   V->data(),  V->ld(),
                                   Aw->data(), Aw->ld(),
                                   static_cast<double>(0.0),
                                   this->data()+
                                    0+basisSize*blockSize*(this->ld()),
                                        this->ld());
            basisSize++;
        }


    }
}

//================================
/*
 * class Subspace::SubspaceHandler
 * */
//================================
template<class fp>
Subspace::SubspaceHandler<fp>::
SubspaceHandler(std::shared_ptr<Subspace::SubspaceBasis<fp>> V)
    : dim{V->Rows()}
    , blockSize{V->getBlockBasisSize()}
    , maxBasisSize{V->getMaxBasisSize()}
    , basisSize{V->getBasisSize()}
    , mgsOrth(dim, blockSize*maxBasisSize)
{
}

template<class fp>
void Subspace::SubspaceHandler<fp>::
orthDirection(
std::shared_ptr<Subspace::SubspaceBasis<fp>> V,
std::shared_ptr<LinearAlgebra::Matrix<fp>> w)
{
    if(w->Cols() < 0) return;

    if(V->Cols()>0)
    {
        if(Orthogonalization::OrthogonalizationErr_t::NO_ERROR !=
                mgsOrth.orthAgainst(dim, V->Cols(), V, w->Cols(), w))
        {
            std::cout << "ERROR!" << std::endl;
        }
    }
    mgsOrth.orth(dim, w->Cols(), w);
}

template<class fp>
void Subspace::SubspaceHandler<fp>::
updateBasis(
std::shared_ptr<Subspace::SubspaceBasis<fp>> V,
std::shared_ptr<LinearAlgebra::Matrix<fp>> w)
{
    if(V->Rows() == w->Rows() && V->Cols()<V->getMaxBasisSize())
    {
        V->insertDirection(w);
        basisSize++;
    }
}

template<class fp>
void Subspace::SubspaceHandler<fp>::
updateProjection(
std::shared_ptr<LinearAlgebra::Matrix<fp>> Aw,
std::shared_ptr<Subspace::SubspaceBasis<fp>> V,
std::shared_ptr<Subspace::SubspaceProjection<fp>> H)
{
    if(Aw->Cols() < 0) return;

    if(V->Cols()>0 && V->Cols()<V->getMaxBasisSize())
    {
        H->updateProjection(V,Aw);
    }
}

template<class fp>
void Subspace::SubspaceHandler<fp>::
restartBasis(std::shared_ptr<Subspace::SubspaceBasis<fp>> V,
             std::shared_ptr<LinearAlgebra::Matrix<fp>> Xprev,
             std::shared_ptr<LinearAlgebra::Matrix<fp>> X,
             std::shared_ptr<LinearAlgebra::Matrix<fp>> w)
{
    V->clear();

    orthDirection(V,Xprev);
    updateBasis(V,Xprev);

    orthDirection(V,X);
    updateBasis(V,X);

    orthDirection(V,w);
    updateBasis(V,w);
}

