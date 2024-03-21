#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <LinearAlgebra/Matrix.hpp>
#include <ProjectionSubspace/Subspace.hpp>

#include <iostream>
#include <memory>

TEST(TestSubspaceHandler, SubspaceHandler) {

    INT rows = 10;
    INT blockCols = 2;
    INT maxCols = 5;

    std::shared_ptr<Subspace::SubspaceBasis<double>>
    V{new Subspace
        ::SubspaceBasis<double>(rows,blockCols,maxCols)};

    Subspace::SubspaceHandler<double> davidsonSubspace(V);

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    w{new LinearAlgebra::Matrix<double>(rows, blockCols)};
    w->rand();

    davidsonSubspace.updateBasis(V,w);

    EXPECT_EQ(blockCols, V->Cols());
    EXPECT_EQ(rows,V->Rows());
    EXPECT_EQ(1,V->getBasisSize());
    w->rand();

    davidsonSubspace.updateBasis(V,w);
    EXPECT_EQ(2*blockCols, V->Cols());
    EXPECT_EQ(rows,V->Rows());
    EXPECT_EQ(2,V->getBasisSize());
}


TEST(TestSubspaceHandler, updateBasis) {

    INT rows = 10;
    INT blockCols = 2;
    INT maxCols = 5;

    std::shared_ptr<Subspace::SubspaceBasis<double>>
    V{new Subspace
        ::SubspaceBasis<double>(rows,blockCols,maxCols)};

    Subspace::SubspaceHandler<double> davidsonSubspace(V);

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    w{new LinearAlgebra::Matrix<double>(rows, blockCols)};
    w->rand();

    // initialize with first direction
    davidsonSubspace.orthDirection(V,w);
    davidsonSubspace.updateBasis(V,w);

    EXPECT_EQ(blockCols, V->Cols());
    EXPECT_EQ(rows,V->Rows());
    EXPECT_EQ(1,V->getBasisSize());

    w->rand();

    // update with seconde direction
    davidsonSubspace.orthDirection(V,w);
    davidsonSubspace.updateBasis(V,w);
    EXPECT_EQ(2*blockCols, V->Cols());
    EXPECT_EQ(rows,V->Rows());
    EXPECT_EQ(2,V->getBasisSize());



    std::shared_ptr<LinearAlgebra::Matrix<double>>
    C{new LinearAlgebra::Matrix<double>(2*blockCols,2*blockCols)};

    LinearAlgebra::Operation::gemm(blas::Layout::ColMajor,
                                   blas::Op::Trans,
                                   blas::Op::NoTrans,
                                   2*blockCols, 2*blockCols, rows,
                                   static_cast<double>(1.0),
                                   V->data(), V->ld(),
                                   V->data(), V->ld(),
                                   static_cast<double>(0.0),
                                   C->data(), C->ld());

    auto iter = LinearAlgebra::max_element(V->begin(), V->end());
    double normInf = *(iter);

    for(uint i=0;i<2*blockCols;i++)
    {
        for(uint j=0;j<2*blockCols;j++)
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



TEST(TestSubspaceHandler, updateProjection) {

    INT rows = 10;
    INT blockCols = 1;
    INT maxCols = 5;

    std::shared_ptr<Subspace::SubspaceBasis<double>>
    V{new Subspace
        ::SubspaceBasis<double>(rows,blockCols,maxCols)};

    std::shared_ptr<Subspace::SubspaceProjection<double>>
    H{new Subspace
        ::SubspaceProjection<double>(blockCols,maxCols)};

    Subspace::SubspaceHandler<double> davidsonSubspace(V);

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    w{new LinearAlgebra::Matrix<double>(rows, blockCols)};
    w->rand();

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Aw{new LinearAlgebra::Matrix<double>(rows, blockCols)};
    Aw->rand();

    davidsonSubspace.orthDirection(V,w);
    davidsonSubspace.updateBasis(V,w);
    davidsonSubspace.updateProjection(Aw,V,H);


    w->rand();
    Aw->rand(); // should be replaced by the matrix vector operator

    davidsonSubspace.orthDirection(V,w);
    davidsonSubspace.updateBasis(V,w);
    davidsonSubspace.updateProjection(Aw,V,H);

    auto iter = LinearAlgebra::max_element(V->begin(), V->end());
    double normInf = *(iter);

    for(uint i=0;i<2*blockCols;i++)
    {
        for(uint j=0;j<2*blockCols;j++)
        {
            if(i>j)
            {
                EXPECT_NEAR(0.0,*(H->data()+i+j*(H->ld())),1e-12*normInf);
            }
        }
    }
}


TEST(TestSubspaceHandler, updateProjectionMaxBasis) {

    INT rows = 10;
    INT blockCols = 1;
    INT maxCols = 5;

    std::shared_ptr<Subspace::SubspaceBasis<double>>
    V{new Subspace
        ::SubspaceBasis<double>(rows,blockCols,maxCols)};

    std::shared_ptr<Subspace::SubspaceProjection<double>>
    H{new Subspace
        ::SubspaceProjection<double>(blockCols,maxCols)};

    Subspace::SubspaceHandler<double> davidsonSubspace(V);

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    w{new LinearAlgebra::Matrix<double>(rows, blockCols)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Aw{new LinearAlgebra::Matrix<double>(rows, blockCols)};

    EXPECT_NE(V->Cols() , V->getMaxBasisSize());

    for(auto j=0; j<maxCols; j++)
    {
        w->rand();
        Aw->rand();
        davidsonSubspace.orthDirection(V,w);
        davidsonSubspace.updateBasis(V,w);
        davidsonSubspace.updateProjection(Aw,V,H);
    }

    EXPECT_EQ(V->Cols() , V->getMaxBasisSize());

    // this new update should not proceed
    w->rand();
    Aw->rand();
    davidsonSubspace.orthDirection(V,w);
    davidsonSubspace.updateBasis(V,w);
    davidsonSubspace.updateProjection(Aw,V,H);
    EXPECT_EQ(V->Cols(), V->getMaxBasisSize());
}

TEST(TestSubspaceHandler, restartBasis) {

    INT rows = 10;
    INT blockCols = 1;
    INT maxCols = 5;

    std::shared_ptr<Subspace::SubspaceBasis<double>>
    V{new Subspace
        ::SubspaceBasis<double>(rows,blockCols,maxCols)};

    std::shared_ptr<Subspace::SubspaceProjection<double>>
    H{new Subspace
        ::SubspaceProjection<double>(blockCols,maxCols)};

    Subspace::SubspaceHandler<double> davidsonSubspace(V);


    std::shared_ptr<LinearAlgebra::Matrix<double>>
    w{new LinearAlgebra::Matrix<double>(rows, blockCols)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Aw{new LinearAlgebra::Matrix<double>(rows, blockCols)};


    EXPECT_NE(V->Cols() , V->getMaxBasisSize());

    for(auto j=0; j<maxCols; j++)
    {
        w->rand();
        Aw->rand();
        davidsonSubspace.orthDirection(V,w);
        davidsonSubspace.updateBasis(V,w);
        davidsonSubspace.updateProjection(Aw,V,H);

    }

    EXPECT_EQ(V->Cols(), V->getMaxBasisSize());
    EXPECT_EQ(H->Cols(), V->getMaxBasisSize());

    // this new update should not proceed
    w->rand();
    Aw->rand();
    davidsonSubspace.orthDirection(V,w);
    davidsonSubspace.updateBasis(V,w);
    davidsonSubspace.updateProjection(Aw,V,H);

    EXPECT_EQ(V->Cols(), V->getMaxBasisSize());
    EXPECT_EQ(H->Cols(), V->getMaxBasisSize());

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    Xprev{new LinearAlgebra::Matrix<double>(rows, blockCols)};

    std::shared_ptr<LinearAlgebra::Matrix<double>>
    X{new LinearAlgebra::Matrix<double>(rows, blockCols)};

    Xprev->rand();
    X->rand();
    w->rand();



    davidsonSubspace.restartBasis(V,H,Xprev, X, w);
    EXPECT_EQ(V->Cols(), 3*blockCols);
    EXPECT_EQ(H->Cols(), 3*blockCols);
    EXPECT_EQ(H->Rows(), 3*blockCols);


    std::shared_ptr<LinearAlgebra::Matrix<double>>
    VTV{new LinearAlgebra::Matrix<double>(V->Cols(), V->Cols())};

    LinearAlgebra::Operation::gemm(blas::Layout::ColMajor,
                                   blas::Op::Trans,
                                   blas::Op::NoTrans,
                                   V->Cols(), V->Cols(), V->Rows(),
                                   static_cast<double>(1.0),
                                   V->data(), V->ld(),
                                   V->data(), V->ld(),
                                   static_cast<double>(0.0),
                                   VTV->data(), VTV->ld());


    auto iter = LinearAlgebra::max_element(V->begin(), V->end());
    double normInf = *(iter);

    for(uint i=0;i<V->Cols();i++)
    {
        for(uint j=0;j<V->Cols();j++)
        {
            if(i==j)
            {
                EXPECT_NEAR(1.0,*(VTV->data()+i+j*(VTV->ld())),1e-12*normInf);
            }
            else
            {
                EXPECT_NEAR(0.0,*(VTV->data()+i+j*(VTV->ld())),1e-12*normInf);
            }
        }
    }

}
