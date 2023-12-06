#include <LinearAlgebra/Matrix.hpp>
#include <Orthogonalizations/MGS.hpp>

#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

TEST(TestMGS, MGS) {

    Orthogonalization::MGS<double> mgsOrth;
}

TEST(TestMGS, QR)
{

    INT m=1000;
    INT n=60;

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

    for(uint i=0;i<m;i++)
    {
        for(uint j=0;j<n;j++)
        {
            EXPECT_DOUBLE_EQ(0.0, *(C->data() + i+j*(C->ld())));
        }
    }

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Q{new LinearAlgebra::Matrix<double>(m,n)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    R{new LinearAlgebra::Matrix<double>(n,n)};

    *Q = *A;
    Orthogonalization::MGS<double> mgsOrth;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::NO_ERROR
              , mgsOrth.QR(m,n,Q,R));

    LinearAlgebra::Operation::gemm(blas::Layout::ColMajor,
                                   blas::Op::NoTrans,
                                   blas::Op::NoTrans,
                                   m, n, n,
                                   static_cast<double>(1.0),
                                   Q->data(), Q->ld(),
                                   R->data(), R->ld(),
                                   static_cast<double>(0.0),
                                   C->data(), C->ld());

    INT index = LinearAlgebra::Operation::iamax(m*n,A->data(),1);
    double normInf = *(A->data()+index);

    for(uint i=0;i<m;i++)
    {
        for(uint j=0;j<n;j++)
        {
            EXPECT_NEAR((*A)[i+j*(A->ld())],(*C)[i+j*(C->ld())],1e-12*normInf);
        }
    }


    Orthogonalization::MGS<double> mgsOrthFailed1;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed1.QR(n-1,n,Q,R));


    Orthogonalization::MGS<double> mgsOrthFailed2;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed2.QR(m,n,NULL,R));

    Orthogonalization::MGS<double> mgsOrthFailed3;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed3.QR(m,n,Q,NULL));

    R->clear();
    Orthogonalization::MGS<double> mgsOrthFailed4;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed4.QR(m,n,Q,R));

    Q->clear();
    Orthogonalization::MGS<double> mgsOrthFailed5;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed5.QR(m,n,Q,R));

}


TEST(TestMGS, orth)
{

    INT m=1000;
    INT n=40;

    /*  A = [ 2 4;
     *        2 5;
     *        2 6;
     *      ]
     */
    std::shared_ptr<LinearAlgebra::Matrix<double>>
    A{new LinearAlgebra::Matrix<double>(m,n)};

    A->rand();


    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Q{new LinearAlgebra::Matrix<double>(m,n)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    C{new LinearAlgebra::Matrix<double>(n,n)};

    *Q = *A;
    Orthogonalization::MGS<double> mgsOrth;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::NO_ERROR
              , mgsOrth.orth(m,n,Q));

    LinearAlgebra::Operation::gemm(blas::Layout::ColMajor,
                                   blas::Op::Trans,
                                   blas::Op::NoTrans,
                                   n, n, m,
                                   static_cast<double>(1.0),
                                   Q->data(), Q->ld(),
                                   Q->data(), Q->ld(),
                                   static_cast<double>(0.0),
                                   C->data(), C->ld());
    INT index = LinearAlgebra::Operation::iamax(m*n,A->data(),1);
    double normInf = *(A->data()+index);
    for(uint i=0;i<n;i++)
    {
        for(uint j=0;j<n;j++)
        {
            if(i!=j)
            {
                EXPECT_NEAR(0.0,(*C)[i+j*(C->ld())],1e-12*normInf);
            }
            else
            {
                EXPECT_NEAR(1.0,(*C)[i+j*(C->ld())],1e-12*normInf);
            }
        }
    }


    Orthogonalization::MGS<double> mgsOrthFailed1;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed1.orth(n-1,n,Q));


    Orthogonalization::MGS<double> mgsOrthFailed2;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed2.orth(m,n,NULL));

    Orthogonalization::MGS<double> mgsOrthFailed3;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed3.orth(n,n,Q));

    Q->clear();
    Orthogonalization::MGS<double> mgsOrthFailed4;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed4.orth(m,n,Q));

}

TEST(TestMGS, orthAgainst)
{

    INT m  = 1000;
    INT nQ = 30;
    INT nW = 10;

    /*  A = [ 2 4;
     *        2 5;
     *        2 6;
     *      ]
     */
    std::shared_ptr<LinearAlgebra::Matrix<double>>
    A{new LinearAlgebra::Matrix<double>(m,nQ)};

    A->rand();


    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Q{new LinearAlgebra::Matrix<double>(m,nQ)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    w{new LinearAlgebra::Matrix<double>(m,nW)};

    w->rand();

    *Q = *A;
    Orthogonalization::MGS<double> mgsOrth;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::NO_ERROR
              , mgsOrth.orth(m,nQ,Q));

    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::NO_ERROR
              , mgsOrth.orth(m,nW,w));

    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::NO_ERROR
              , mgsOrth.orthAgainst(m,nQ,Q,nW,w));


    INT index = LinearAlgebra::Operation::iamax(m*nQ, Q->data(),1);
    double normInf = *(Q->data()+index);

    for(uint i=0;i<nQ;i++)
    {
        for(uint j=0;j<nW;j++)
        {
            auto dist = LinearAlgebra::Operation::dot(m,
                                                      w->data()+0+j*(w->ld()),
                                                      1,
                                                      Q->data()+0+i*(Q->ld()),
                                                      1);
        EXPECT_NEAR(0.0, dist ,1e-12*normInf);
        }
    }

    Orthogonalization::MGS<double> mgsOrthFailed1;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed1.orthAgainst(nQ-1,nQ,Q,nW,w));

    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed1.orthAgainst(nW-1,nQ,Q,nW,w));

    Orthogonalization::MGS<double> mgsOrthFailed2;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed2.orthAgainst(m,nQ,NULL,nW,w));

    Orthogonalization::MGS<double> mgsOrthFailed3;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed3.orthAgainst(m,nQ,Q,nW,NULL));

    w->clear();
    Orthogonalization::MGS<double> mgsOrthFailed4;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed4.orthAgainst(nQ,nQ,Q,nW,w));

    Q->clear();
    Orthogonalization::MGS<double> mgsOrthFailed5;
    EXPECT_EQ(Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT
              , mgsOrthFailed5.orthAgainst(nW,nQ,Q,nW,w));

}
