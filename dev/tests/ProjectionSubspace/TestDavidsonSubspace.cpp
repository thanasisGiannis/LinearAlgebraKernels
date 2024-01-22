#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <LinearAlgebra/Matrix.hpp>
#include <ProjectionSubspace/DavidsonSubspace.hpp>

#include <iostream>
#include <memory>

TEST(TestDavidsonSubspace, DavidsonSubspace) {

    ProjectionSubspace::DavidsonSubspace<double> davidsonSubspace(10,1,5);
    EXPECT_EQ(1,davidsonSubspace.getRawBasisSize());
}

TEST(TestDavidsonSubspace, updateBasis) {

    INT dim=10;
    INT blockSize = 3;
    INT maxBasisSize = 15;
    ProjectionSubspace::DavidsonSubspace<double> davidsonSubspace(dim,blockSize,maxBasisSize);
    EXPECT_EQ(1*blockSize,davidsonSubspace.getRawBasisSize());

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    v{new LinearAlgebra::Matrix<double>(dim,blockSize)};
    v->rand();

    davidsonSubspace.updateBasis(v);
    EXPECT_EQ(2*blockSize,davidsonSubspace.getRawBasisSize());

    auto Q = davidsonSubspace.getRawBasisMatrix();

    auto m = dim;
    auto n = Q->Cols();

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    C{new LinearAlgebra::Matrix<double>(n,n)};


    LinearAlgebra::Operation::gemm(blas::Layout::ColMajor,
                                   blas::Op::Trans,
                                   blas::Op::NoTrans,
                                   n, n, m,
                                   static_cast<double>(1.0),
                                   Q->data(), Q->ld(),
                                   Q->data(), Q->ld(),
                                   static_cast<double>(0.0),
                                   C->data(), C->ld());

    auto iter = LinearAlgebra::max_element(Q->begin(), Q->end());
    double normInf = *(iter);

    for(uint i=0;i<n;i++)
    {
        for(uint j=0;j<n;j++)
        {
            if(i!=j)
            {
                EXPECT_NEAR(0.0,*(C->data()+i+j*(C->ld())),1e-12*normInf);
            }
            else
            {
                EXPECT_NEAR(1.0,*(C->data()+i+j*(C->ld())),1e-12*normInf);
            }
        }
    }
}
