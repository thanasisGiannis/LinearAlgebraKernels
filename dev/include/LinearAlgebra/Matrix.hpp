#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <LinearAlgebra/Operations.hpp>

#include <vector>
#include <list>
#include <iterator>
#include <memory>
#include <iostream>
#include <functional>
#include <algorithm>
#include <random>

namespace LinearAlgebra {

/*
 * Column Major Dense Matrix Representation
 *
 */
template<class fp>
class Matrix
{
private:
    INT rows;           // number of rows and leading dimension
    INT cols;           // number of columns
    vector<fp> raw_data; // raw data

public:
    Matrix(INT rows_, INT cols_)
        : rows{rows_}
        , cols{cols_}
    {
        raw_data.resize(rows*cols);
        LinearAlgebra::fill(raw_data.begin(),
                            raw_data.end(),
                            static_cast<fp>(0.0));
    }

    Matrix(INT rows_, INT cols_, std::list<fp> l)
        : rows{rows_}
        , cols{cols_}
    {
        if( static_cast<INT>(l.size()) == rows_*cols_)
        {
            for(fp item : l)
            {
                raw_data.push_back(item);
            }
            l.clear();
        }
        if(raw_data.size()!=rows_*cols_)
        {
            raw_data.resize(rows_*cols_);
        }
    }

    auto size(){return static_cast<INT>(raw_data.size());}
    auto begin(){return raw_data.begin();}
    auto end()  {return raw_data.end();}
    INT ld(){return rows;}
    fp* data()
    {

        #if defined( BLAS_HAVE_CUBLAS ) \
            || defined( BLAS_HAVE_ROCBLAS ) \
            || defined( BLAS_HAVE_SYCL )
            return static_cast<fp*>(raw_pointer_cast(raw_data.data()));
        #else
            raw_data.data();
        #endif
    }
    auto operator[](INT index) {return raw_data[index];}
    auto Rows(){return rows;}
    auto Cols(){return cols;}
    auto clear()
    {
        rows = 0;
        cols=0;
        return raw_data.clear();
    }

    void rand()
    {


        host_vector<fp> tmp(rows*cols);
        int range = static_cast<int>(this->rows+this->cols);
        // First create an instance of an engine.
        std::random_device rnd_device;
        // Specify the engine and distribution.
        std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
        std::uniform_int_distribution<int> dist {-range, range};

        auto gen = [&dist, &mersenne_engine,&range](){
                       return static_cast<fp>(dist(mersenne_engine)/2*range);
                   };

        LinearAlgebra::generate(tmp.begin(), tmp.end(),
                 [&range, &gen](){
                    auto r = gen()/(range*range);
                    return r;
                 });

        for(INT i=0;i<this->cols;i++)
        {
            tmp[i+i*this->ld()]
                    = std::abs(tmp[i+i*this->ld()])
                    + static_cast<fp>(this->cols);
        }
        raw_data = tmp;
    }
};

} // namespace LinearAlgebra

template<class fp>
std::ostream& operator<<(std::ostream&os , LinearAlgebra::Matrix<fp> &A)
{
    auto m   = A.Rows();
    auto n   = A.Cols();
    auto ldA = A.ld();
    os << "=[ " << std::endl;
    for(auto i=0; i<m; i++)
    {
        for(auto j=0; j<n; j++)
        {
            os << A[i+j*ldA] << " ";
        }
        os << " ; " << std::endl;
    }
    os << "];" << std::endl;;
    return os;
}

#endif // MATRIX_HPP
