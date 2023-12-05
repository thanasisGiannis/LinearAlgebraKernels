#include <string.h>

template<class fp>
Orthogonalization::Householder<fp>::
Householder(INT dim, INT nrhs)
    : rowMaxDim{dim}
    , colMaxDim{nrhs}
    , hhQz{LinearAlgebra::Matrix<fp>(dim,dim)}
    , hhx{LinearAlgebra::Matrix<fp>(dim,1)}
    , hhv{LinearAlgebra::Matrix<fp>(dim,1)}
    , hhu{LinearAlgebra::Matrix<fp>(nrhs,1)}
    , hhvhhvt{LinearAlgebra::Matrix<fp>(dim,dim)}
{
}

template<class fp>
Orthogonalization::OrthogonalizationErr_t
Orthogonalization::Householder<fp>::
QR(INT m, INT n,
   std::shared_ptr<LinearAlgebra::Matrix<fp>> Q_,
   std::shared_ptr<LinearAlgebra::Matrix<fp>> R_)
{
    if( m<n               ||
        m > rowMaxDim     ||
        n > colMaxDim     ||
        NULL == Q_        ||
        NULL == R_        ||
        0 >  Q_->size()   ||
        0 >  R_->size()   ||
        m*m <  Q_->size() ||
        m*n <  R_->size() ||
         0  == Q_->size() ||
         0  == R_->size() )
    {
        return OrthogonalizationErr_t::INVALID_INPUT;
    }

    LinearAlgebra::Matrix<fp> &Q = *Q_; INT ldQ = Q.ld();
    LinearAlgebra::Matrix<fp> &R = *R_; INT ldR = R.ld();
    auto ldhhQz     = hhQz.ld();
    auto ldhhv      = hhv.ld();
    auto ldhhu      = hhu.ld();
    auto ldhhvhhvt  = hhvhhvt.ld();

    // Q = I
    fill(Q.begin(), Q.end(), static_cast<fp>(0.0));

    for(INT i=0; i<m; i++){
      Q[i+i*ldQ] = static_cast<fp>(1.0);
    }

    for(INT k=0; k<n; k++) {
      // x = zeros(m,1);
      fill(hhx.begin(), hhx.end(), static_cast<fp>(0.0));

      // x(k:m,1)=R(k:m,k);
      LinearAlgebra::Operation::copy(m-k,&(R[k+k*ldR]),1,&(hhx[k]),1);

      //g=norm(x);
      fp g = LinearAlgebra::Operation::nrm2(m, hhx.data(), 1);

      // v=x; v(k)=x(k)+g;
      hhv = hhx;
      hhv[k] = hhx[k] + g;

      // s=norm(v);
      fp s = LinearAlgebra::Operation::nrm2(m,hhv.data(),1);

      // if s!=0
      // v = v/s
      LinearAlgebra::Operation::scal(m, static_cast<fp>(1.0)/s, hhv.data(),1);

      // u=R'*v;
      LinearAlgebra::Operation::gemm(LinearAlgebra::Operation::Layout::ColMajor,
                                     LinearAlgebra::Operation::Op::Trans,
                                     LinearAlgebra::Operation::Op::NoTrans,
        n, 1, m, static_cast<fp>(1.0), R.data(), ldR,
        hhv.data(), ldhhv , static_cast<fp>(0.0), hhu.data(), ldhhu);

      // R=R-2*v*u';
      LinearAlgebra::Operation::gemm(LinearAlgebra::Operation::Layout::ColMajor,
                                     LinearAlgebra::Operation::Op::NoTrans,
                                     LinearAlgebra::Operation::Op::Trans,
        m, n, 1, static_cast<fp>(-2.0), hhv.data(), m,
        hhu.data(), n, static_cast<fp>(1.0), R.data(), ldR);


      // Q=Q-2*Q*(v*v');
      hhQz = Q;
      //hhvhhvt = v*v';
      LinearAlgebra::Operation::gemm(LinearAlgebra::Operation::Layout::ColMajor,
                                     LinearAlgebra::Operation::Op::NoTrans,
                                     LinearAlgebra::Operation::Op::Trans,
        m, m, 1, static_cast<fp>(1.0), hhv.data(), m,
        hhv.data(), m, static_cast<fp>(0.0), hhvhhvt.data(), ldhhvhhvt);


      // Q = Q-2*Q*hhvhhvt
      LinearAlgebra::Operation::gemm(LinearAlgebra::Operation::Layout::ColMajor,
                                     LinearAlgebra::Operation::Op::NoTrans,
                                     LinearAlgebra::Operation::Op::NoTrans,
                                     m, m, m,
      static_cast<fp>(-2.0), hhQz.data(), ldhhQz, hhvhhvt.data(), ldhhvhhvt,
      static_cast<fp>(1.0), Q.data(), ldQ);

    }
    return OrthogonalizationErr_t::NO_ERROR;
}
