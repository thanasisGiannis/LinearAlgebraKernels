template<class fp>
Orthogonalization::Cholesky<fp>::
Cholesky(const INT dim, const INT nrhs)
    : rowMaxDim{dim}
    , colMaxDim{nrhs}
{

}

template<class fp>
Orthogonalization::OrthogonalizationErr_t
Orthogonalization::Cholesky<fp>::
QR(const INT m, const INT n,
   std::shared_ptr<LinearAlgebra::Matrix<fp>> Q,
   std::shared_ptr<LinearAlgebra::Matrix<fp>> R)
{


    if( m<n               ||
        m > rowMaxDim     ||
        n > colMaxDim     ||
        NULL == Q         ||
        NULL == R         ||
        0 >  Q->size()    ||
        0 >  R->size()    ||
        m*n <  Q->size()  ||
        n*n <  R->size()  ||
         0  == Q->size()  ||
         0  == R->size()  )
    {
        return OrthogonalizationErr_t::INVALID_INPUT;
    }

    // R : n x n // upper triangular
    // Q : m x n

    // R = Q'*Q;
    LinearAlgebra::Operation::gemm(LinearAlgebra::Operation::Layout::ColMajor,
                                    LinearAlgebra::Operation::Op::Trans,
                                    LinearAlgebra::Operation::Op::NoTrans,
                                    n, n, m,
                                    static_cast<fp>(1.0),
                                    Q->data(), Q->ld(),
                                    Q->data(), Q->ld(),
                                    static_cast<fp>(0.0),
                                    R->data(), R->ld());

    // R = chol(R) = chol(Q'*Q)
    LinearAlgebra::Operation::potrf(LinearAlgebra::Operation::Uplo::Upper, n,
                                    R->data(), R->ld());

    // Q = Q/R;
    LinearAlgebra::Operation::trsm(LinearAlgebra::Operation::Layout::ColMajor,
                                    LinearAlgebra::Operation::Side::Right,
                                    LinearAlgebra::Operation::Uplo::Upper,
                                    LinearAlgebra::Operation::Op::NoTrans,
                                    LinearAlgebra::Operation::Diag::NonUnit,
                                    m, n,
                                    static_cast<fp>(1.0),
                                    R->data(), R->ld(),
                                    Q->data(), Q->ld());

    return Orthogonalization::OrthogonalizationErr_t::NO_ERROR;
}
