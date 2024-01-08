
template<class fp>
Orthogonalization::MGS<fp>::MGS(INT m, INT n)
    : rowMaxDim{m}
    , colMaxDim{n}
{
    /*
    size_t d_size, h_size;
    lapack::geqrf_work_size_bytes( m, n, dA_tst, lda, &d_size, &h_size, queue );
    char* d_work = blas::device_malloc< char >( d_size, queue );
    std::vector<char> h_work_vector( h_size );
    char* h_work = h_work_vector.data();
    */
}

template<class fp>
Orthogonalization::OrthogonalizationErr_t
Orthogonalization::MGS<fp>::
QR(const INT m, const INT n,
   std::shared_ptr<LinearAlgebra::Matrix<fp>> Q,
   std::shared_ptr<LinearAlgebra::Matrix<fp>> R)
{
    /*
        [~,n] = size(Q);
        R = zeros(n);
        for j=1:n
          r = norm(Q(:,j));
          R(j,j) = r;
          Q(:,j) = Q(:,j)/r;
          for k=j+1:n
             r = Q(:,j)'*Q(:,k);
             R(j,k) = r;
             Q(:,k) = Q(:,k) - r*Q(:,j);
          end
        end
    */

    if( m<n               ||
        NULL == Q         ||
        NULL == R         ||
        0 >=  Q->size()   ||
        0 >=  R->size()   ||
        m*m < Q->size()  ||
        m*n < R->size()  ||
         0  == Q->size()  ||
         0  == R->size()  )
    {
        return OrthogonalizationErr_t::INVALID_INPUT;
    }

    LinearAlgebra::fill(R->begin(), R->end(), static_cast<fp>(0.0));
    for(int j=0; j<n; j++)
    {
        fp r = LinearAlgebra::Operation::nrm2(m,Q->data() + 0 +j*(Q->ld()), 1);
        *(R->data()+j+j*(R->ld())) = r;
        LinearAlgebra::Operation::scal(m, static_cast<fp>(1.0)/r,
                                       Q->data() + 0 + j*(Q->ld()), 1);
        for(auto k=j+1; k<n; k++)
        {
            r = LinearAlgebra::Operation::dot(m,
                                              Q->data() + 0 + j*(Q->ld()), 1,
                                              Q->data() + 0 + k*(Q->ld()), 1);
            *(R->data()+j+k*(R->ld())) = r;
            LinearAlgebra::Operation::axpy(m, -r,
                                           Q->data() + 0 + j*(Q->ld()), 1,
                                           Q->data() + 0 + k*(Q->ld()), 1);
        }
    }
    return Orthogonalization::OrthogonalizationErr_t::NO_ERROR;
}




template<class fp>
Orthogonalization::OrthogonalizationErr_t
Orthogonalization::MGS<fp>::
orth(const INT m, const INT n,std::shared_ptr<LinearAlgebra::Matrix<fp>> Q)
{
    /*
        [~,n] = size(Q);
        R = zeros(n);
        for j=1:n
          r = norm(Q(:,j));
          Q(:,j) = Q(:,j)/r;
          for k=j+1:n
             r = Q(:,j)'*Q(:,k);
             Q(:,k) = Q(:,k) - r*Q(:,j);
          end
        end
    */

    if( m<n               ||
        NULL == Q         ||
        0 >=  Q->size()   ||
        m*m <  Q->size()  )
    {
        return OrthogonalizationErr_t::INVALID_INPUT;
    }

    for(int j=0; j<n; j++)
    {
        fp r = LinearAlgebra::Operation::nrm2(m, Q->data() + 0 +j*(Q->ld()), 1);
        LinearAlgebra::Operation::scal(m,static_cast<fp>(1.0)/r,
                                       Q->data() + 0 + j*(Q->ld()), 1);

        for(auto k=j+1; k<n; k++)
        {
            r = LinearAlgebra::Operation::dot(m,
                                              Q->data() + 0 +j*(Q->ld()), 1,
                                              Q->data() + 0 + k*(Q->ld()),1);

            LinearAlgebra::Operation::axpy(m, -r,
                                           Q->data() + 0 + j*(Q->ld()), 1,
                                           Q->data() + 0 + k*(Q->ld()), 1);
        }
    }
    return Orthogonalization::OrthogonalizationErr_t::NO_ERROR;
}


template<class fp>
Orthogonalization::OrthogonalizationErr_t
Orthogonalization::MGS<fp>::
orthAgainst(const INT m,
            const INT nQ,
            std::shared_ptr<LinearAlgebra::Matrix<fp>> Q,
            const INT nW,
            std::shared_ptr<LinearAlgebra::Matrix<fp>> W)
{

    /*
        w = orth(V,w);
    */

    if( m<nQ              ||
        m<nW              ||
        NULL == Q         ||
        NULL == W         ||
        0 >  Q->size()    ||
        0 >  W->size()    ||
        m*nQ <  Q->size() ||
        m*nW <  W->size() ||
         0  == Q->size()  ||
         0  == W->size()  )
    {
        return OrthogonalizationErr_t::INVALID_INPUT;
    }

    for(int j=0; j<nW; j++)
    {
        for(int i=0; i<nQ; i++)
        {
            // alpha = Q(:,i)'w(:,j);
            fp alpha = LinearAlgebra::Operation::dot(m,
                                                     Q->data()+0+i*(Q->ld()),
                                                     1,
                                                     W->data()+0+j*(W->ld()),
                                                     1);
            // w(:,j) = w(:,j) - Q(:,i)*alpha;
            LinearAlgebra::Operation::axpy(m, -alpha,
                                           Q->data()+0+i*(Q->ld()), 1,
                                           W->data()+0+j*(W->ld()), 1);
        }
    }
    return Orthogonalization::OrthogonalizationErr_t::NO_ERROR;
}
