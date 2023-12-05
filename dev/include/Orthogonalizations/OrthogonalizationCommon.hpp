#ifndef ORTHOGONALIZATIONCOMMON_HPP
#define ORTHOGONALIZATIONCOMMON_HPP

#include <iostream>

namespace Orthogonalization {
enum OrthogonalizationErr_t
{
    NO_ERROR,
    INVALID_INPUT
};
} // namespace Orthogonalization


template<class fp>
std::ostream& operator<<(std::ostream&os ,
                         Orthogonalization::OrthogonalizationErr_t &err)
{
    switch (err)
    {
    case Orthogonalization::OrthogonalizationErr_t::NO_ERROR:
        os << std::endl << "Householder: NO_ERROR" <<std::endl;
    break;
    case Orthogonalization::OrthogonalizationErr_t::INVALID_INPUT:
        os << std::endl << "Householder: INVALID_INPUT" <<std::endl;
    break;
    }
    return os;
}

#endif // ORTHOGONALIZATIONCOMMON_HPP
