#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <LinearAlgebra/Matrix.hpp>
#include <Orthogonalizations/Cholesky.hpp>

#include <iostream>
#include <memory>

TEST(TestCholesky, Cholesky) 
{

    Orthogonalization::Cholesky<double> cholOrth(100,100);
    EXPECT_EQ(0,0);
}

TEST(TestCholesky, QR)
{

    INT m=1000;
    INT n=21;

    /*  A = [ 2 4;
     *        2 5;
     *        2 6;
     *      ]
     */
    std::shared_ptr<LinearAlgebra::Matrix<double>>
    A{new LinearAlgebra::Matrix<double>(m,n)};

    A->rand();

    auto iter = LinearAlgebra::max_element(A->begin(), A->end());
    double normInf = *(iter);

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    C{new LinearAlgebra::Matrix<double>(m,n)};

    for(uint i=0;i<m;i++)
    {
        for(uint j=0;j<n;j++)
        {
            EXPECT_NEAR(0.0,(*C)[i+j*(C->ld())],1e-12*normInf);
        }
    }

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Q{new LinearAlgebra::Matrix<double>(m,n)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    R{new LinearAlgebra::Matrix<double>(n,n)};

    *Q = *A;
    Orthogonalization::Cholesky<double> cholOrth(m,n);
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::NO_ERROR
              , cholOrth.QR(m,n,Q,R));

    *C = *Q;
    LinearAlgebra::Operation::trmm(LinearAlgebra::Operation::Layout::ColMajor,
                                    LinearAlgebra::Operation::Side::Right,
                                    LinearAlgebra::Operation::Uplo::Upper,
                                    LinearAlgebra::Operation::Op::NoTrans,
                                    LinearAlgebra::Operation::Diag::NonUnit,
                                    m,n,1.0,R->data(), R->ld(),C->data(),C->ld());

    for(uint i=0;i<m;i++)
    {
        for(uint j=0;j<n;j++)
        {
            EXPECT_NEAR(0,std::abs((*A)[i+j*(A->ld())]-(*C)[i+j*(C->ld())]),
                        1e-12*normInf);
        }
    }
}

#if 0
TEST(TestCholesky, qrBigMatrix)
{

    INT m=1000;
    INT n=101;

    /*  A = [ 2 4;
     *        2 5;
     *        2 6;
     *      ]
     */
    std::shared_ptr<LinearAlgebra::Matrix<double>>
    A{new LinearAlgebra::Matrix<double>(m,n)};

    A->rand();

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    C{new LinearAlgebra::Matrix<double>(m,n)};

    auto iter = LinearAlgebra::max_element(A->begin(), A->end());
    double normInf = *(iter);

    for(uint i=0;i<m;i++)
    {
        for(uint j=0;j<n;j++)
        {
            EXPECT_NEAR(0.0,(*C)[i+j*(C->ld())],1e-12*normInf);
        }
    }

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Q{new LinearAlgebra::Matrix<double>(m,n)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    R{new LinearAlgebra::Matrix<double>(n,n)};

    *Q = *A;
    Orthogonalization::Cholesky<double> cholOrth(m,n);
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::NO_ERROR
              , cholOrth.QR(m,n,Q,R));

    *C = *Q;
    LinearAlgebra::Operation::trmm(LinearAlgebra::Operation::Layout::ColMajor,
                                    LinearAlgebra::Operation::Side::Right,
                                    LinearAlgebra::Operation::Uplo::Upper,
                                    LinearAlgebra::Operation::Op::NoTrans,
                                    LinearAlgebra::Operation::Diag::NonUnit,
                                    m,n,1.0,R->data(), R->ld(),C->data(),C->ld());


    for(uint i=0;i<m;i++)
    {
        for(uint j=0;j<n;j++)
        {
            EXPECT_NEAR(0,(*A)[i+j*(A->ld())]-(*C)[i+j*(C->ld())],
                        1e-12*normInf);
        }
    }
}
#endif
