template<class fp>
ProjectionSubspace::DavidsonSubspace<fp>::
DavidsonSubspace(INT dim_, INT blockSize_, INT maxBasisSize_)
    : dim{dim_}
    , blockSize{blockSize_<=dim?blockSize_:dim_}
    , maxBasisSize{(blockSize*maxBasisSize_)<=dim?maxBasisSize:((INT)floor(dim/blockSize))}
    , basisSize{0}
    , V{new LinearAlgebra::Matrix<fp>(dim,1*blockSize)}
    , mgsOrth(dim, blockSize*maxBasisSize)
{
    V->reserveCols(maxBasisSize*blockSize);
    basisSize++;
    V->resizeCols(basisSize*blockSize);
    V->rand();
    mgsOrth.orth(dim, basisSize*blockSize, V);
}

template<class fp>
std::shared_ptr<LinearAlgebra::Matrix<fp>> ProjectionSubspace
::DavidsonSubspace<fp>::getRawBasisMatrix()
{
    return V;
}

template<class fp>
INT ProjectionSubspace::DavidsonSubspace<fp>::
getRawBasisSize()
{
    return blockSize*basisSize;
}

template<class fp>
INT ProjectionSubspace::DavidsonSubspace<fp>::
getMaxBasisSize()
{
   return maxBasisSize;
}

template<class fp>
INT ProjectionSubspace::DavidsonSubspace<fp>::
getBlockBasisSize()
{
   return blockSize;
}

template<class fp>
void ProjectionSubspace::DavidsonSubspace<fp>::
updateBasis(std::shared_ptr<LinearAlgebra::Matrix<fp>> w)
{
    if(w->Cols() == blockSize)
    {
        if(Orthogonalization::OrthogonalizationErr_t::NO_ERROR !=
                mgsOrth.orthAgainst(dim, V->Cols(), V, w->Cols(), w))
        {
            std::cout << "ERROR!" << std::endl;
        }

        mgsOrth.orth(dim, w->Cols(), w);
        V->insertCols(w);
        basisSize++;
    }
}
