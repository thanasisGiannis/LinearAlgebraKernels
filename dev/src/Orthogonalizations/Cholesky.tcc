template<class fp>
Orthogonalization::Cholesky<fp>::
Cholesky(const INT dim, const INT nrhs)
    : rowMaxDim{dim}
    , colMaxDim{nrhs}
    , B{LinearAlgebra::Matrix<fp>(nrhs,nrhs)}
{

}

template<class fp>
bool Orthogonalization::Cholesky<fp>::
chol(const int n, std::shared_ptr<LinearAlgebra::Matrix<fp>> L)
{
    /*
     * Equivelant of matlab's
     * L = chol(B);
     */

    /*
     * ToDo: this may need to bee chaned into
     * a more GPU friendly function
     */

    /*
      [n,~] = size(A);
      for j=1:n
          sum = 0;
          for k=1:j
              sum = sum + L(k,j)*L(k,j);
          end
          L(j,j) = sqrt(A(j,j)-sum);

          for i=j+1:n
              sum = 0;
              for k=1:j
                 sum = sum + L(k,i)*L(k,j);
              end
              L(j,i) = (1/L(j,j))*(A(j,i)-sum);
          end
      end
    */

    // L = zeros(n);
    fill(L->begin(), L->end(), static_cast<fp>(0.0));

    fp sum;
    for(int j=0; j<n; j++)
    {
        sum = static_cast<fp>(0.0);
        for(int k=0; k<j; k++)
        {
            sum = sum + (*(L->data()+k+j*(L->ld())))*(*(L->data()+k+j*(L->ld())));
        }

        *(L->data()+j+j*(L->ld()))= sqrt(*(B.data()+j+j*(B.ld()))-sum);

        if(std::isnan(*(L->data()+j+j*(L->ld())))) return false;

        for(int i=j+1; i<n; i++)
        {
            sum = static_cast<fp>(0.0);
            for(int k=0; k<j; k++)
            {
                sum = sum + *(L->data()+k+i*(L->ld()))*(*(L->data()+k+j*(L->ld())));
            }

            *(L->data()+j+i*(L->ld()))
                    = (static_cast<fp>(1.0)/(*(L->data()+j+j*(L->ld())))*(*(B.data()+j+i*(B.ld()))-sum));

            if(std::isnan(*(L->data()+j+i*(L->ld())))) return false;
        }
    }
    return true;
}

template<class fp>
Orthogonalization::OrthogonalizationErr_t
Orthogonalization::Cholesky<fp>::
QR(const INT m, const INT n,
   std::shared_ptr<LinearAlgebra::Matrix<fp>> Q,
   std::shared_ptr<LinearAlgebra::Matrix<fp>> R)
{


    // R : n x n
    // Q : m x n
    fill(B.begin(), B.end(), static_cast<fp>(0.0));
    // B = Q'*Q;
    LinearAlgebra::Operation::gemm(LinearAlgebra::Operation::Layout::ColMajor,
                                    LinearAlgebra::Operation::Op::Trans,
                                    LinearAlgebra::Operation::Op::NoTrans,
                                    n, n, m,
                                    static_cast<fp>(1.0),
                                    Q->data(), Q->ld(),
                                    Q->data(), Q->ld(),
                                    static_cast<fp>(0.0),
                                    B.data(), B.ld());

    if(false == chol(n, R))
    {
        return Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT;
    }

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
