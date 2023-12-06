#include <LinearAlgebra/Matrix.hpp>
#include <Orthogonalizations/Householder.hpp>

#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

TEST(TestHouseholder, Householder) {

    Orthogonalization::Householder<double> hsOrth(100,100);
}

TEST(TestHouseholder, QR)
{

    INT m=3;
    INT n=2;

    /*  A = [ 2 4;
     *        2 5;
     *        2 6;
     *      ]
     */
    std::shared_ptr<LinearAlgebra::Matrix<double>>
    A{new LinearAlgebra::Matrix<double>(m,n,{2,2,2,
                                             4,5,6})};

    INT index = LinearAlgebra::Operation::iamax(m*n,A->data(),1);
    double normInf = *(A->data()+index);

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    C{new LinearAlgebra::Matrix<double>(m,n)};

    for(uint i=0;i<m;i++)
    {
        for(uint j=0;j<n;j++)
        {
            EXPECT_NEAR(0.0,*(C->data()+i+j*(C->ld())),1e-12*normInf);
        }
    }

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Q{new LinearAlgebra::Matrix<double>(m,m)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    R{new LinearAlgebra::Matrix<double>(m,n)};

    *R = *A;
    Orthogonalization::Householder<double> hs(m,n);
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::NO_ERROR
              , hs.QR(m,n,Q,R));

    LinearAlgebra::Operation::gemm(blas::Layout::ColMajor,
                                   blas::Op::NoTrans,
                                   blas::Op::NoTrans,
                                   m, n, m,
                                   static_cast<double>(1.0),
                                   Q->data(), Q->ld(),
                                   R->data(), R->ld(),
                                   static_cast<double>(0.0),
                                   C->data(), C->ld());

    for(uint i=0;i<m;i++)
    {
        for(uint j=0;j<n;j++)
        {
            EXPECT_NEAR(*(A->data()+i+j*(A->ld())),
                        *(C->data()+i+j*(C->ld())),1e-12*normInf);
        }
    }
}

TEST(TestHouseholder, checkInputQR)
{

    INT m=3;
    INT n=2;

    /*  A = [ 2 4;
     *        2 5;
     *        2 6;
     *      ]
     */
    std::shared_ptr<LinearAlgebra::Matrix<double>>
    A{new LinearAlgebra::Matrix<double>(m,n,{2,2,2,
                                             4,5,6})};

    /*  A = [ 2 4 3;
     *        2 5 3;
     *        2 6 5;
     *      ]
     */
    std::shared_ptr<LinearAlgebra::Matrix<double>>
    B{new LinearAlgebra::Matrix<double>(m,m,{2,2,2,
                                             4,5,6,
                                             3,3,5})};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Q{new LinearAlgebra::Matrix<double>(m,m)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    R{new LinearAlgebra::Matrix<double>(m,n)};

    *R = *A;
    Orthogonalization::Householder<double> hsCase1(m-1,n-1);

    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT,
              hsCase1.QR(m,n,Q,R));

    *R = *A;
    Orthogonalization::Householder<double> hsCase2(m,n);

    Q->clear();
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT,
              hsCase2.QR(m,n,Q,R));

    *Q=*B;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT,
              hsCase2.QR(m,n,Q,NULL));

    *Q=*B;
    *R=*B;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT,
              hsCase2.QR(m,n,Q,R));
}

TEST(TestHouseholder, checkBigMatrixQR)
{

    INT m=100;
    INT n=40;

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    A{new LinearAlgebra::Matrix<double>(m,n)};

    A->rand();

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Q{new LinearAlgebra::Matrix<double>(m,m)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    R{new LinearAlgebra::Matrix<double>(m,n)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    C{new LinearAlgebra::Matrix<double>(m,n)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    I{new LinearAlgebra::Matrix<double>(m,m)};

    *R = *A;
    Orthogonalization::Householder<double> hs(m,n);
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::NO_ERROR
              , hs.QR(m,n,Q,R));


    LinearAlgebra::Operation::gemm(blas::Layout::ColMajor,
                                   blas::Op::NoTrans,
                                   blas::Op::NoTrans,
                                   m, n, m,
                                   static_cast<double>(1.0),
                                   Q->data(), Q->ld(),
                                   R->data(), R->ld(),
                                   static_cast<double>(0.0),
                                   C->data(), C->ld());

    INT index = LinearAlgebra::Operation::iamax(m*n,A->data(),1);
    double normInf = *(A->data()+index);


    // check correctness of values
    // [Q,R] = qr(A);
    // C = Q*R;
    // A-C < 1e-12*norm(A,'inf');
    for(uint i=0;i<m;i++)
    {
        for(uint j=0;j<n;j++)
        {
            EXPECT_NEAR(*(A->data()+i+j*(A->ld())),
                        *(C->data()+i+j*(C->ld())),
                        1e-12*normInf);
        }
    }

    LinearAlgebra::Operation::gemm(blas::Layout::ColMajor,
                                   blas::Op::Trans,
                                   blas::Op::NoTrans,
                                   m, m, m,
                                   static_cast<double>(1.0),
                                   Q->data(), Q->ld(),
                                   Q->data(), Q->ld(),
                                   static_cast<double>(0.0),
                                   I->data(), I->ld());


    // check if I-Q'*Q < 1e-12*norm(A,'inf');
    for(uint i=0;i<m;i++)
    {
        for(uint j=0;j<n;j++)
        {
            if(i==j)
            {
                EXPECT_NEAR(1.0,*(I->data()+i+j*(I->ld())), 1e-12*normInf);
            }
            else
            {
                EXPECT_NEAR(0.0,*(I->data()+i+j*(I->ld())), 1e-12*normInf);
            }
        }
    }

    // check if R is upper triangular
    for(uint j=0;j<n;j++)
    {
        for(uint i=j+1;i<m;i++)
        {
            EXPECT_NEAR(0.0,*(R->data()+i+j*(R->ld())), 1e-12*normInf);
        }
    }

}
