#ifndef RVTX_MOLECULE_COLOR_HPP
#define RVTX_MOLECULE_COLOR_HPP

#include <glm/vec3.hpp>

namespace rvtx
{
    struct Atom;
    struct Chain;

    glm::vec3 getAtomColor( const rvtx::Atom & atom );
    glm::vec3 getChainColor( const rvtx::Chain & chain );
} // namespace rvtx

#endif // RVTX_MOLECULE_COLOR_HPP
