#ifndef RVTX_SYSTEM_MOLECULE_IDS_HPP
#define RVTX_SYSTEM_MOLECULE_IDS_HPP

#include <array>

#include "glm/vec4.hpp"
#include "rvtx/core/type.hpp"

namespace rvtx
{
    class Molecule;
    class Atom;
    class Bond;

    struct MoleculeIDs
    {
        static constexpr bool in_place_delete = true;

        explicit MoleculeIDs( Molecule & moleculeRef );

        inline bool contains( const uint32_t id ) const;

        inline std::size_t atomIndex( const uint32_t id ) const;
        inline std::size_t bondIndex( const uint32_t id ) const;

        inline Atom      getAtom( const uint32_t id ) const;
        inline glm::vec4 getData( const uint32_t id ) const;

        inline Bond                     getBond( const uint32_t id ) const;
        inline std::array<Atom, 2>      getBondAtoms( const uint32_t id ) const;
        inline std::array<glm::vec4, 2> getBondData( const uint32_t id ) const;

        Range atomIds {};
        Range bondIds {};

        Molecule * molecule = nullptr;
    };
} // namespace rvtx

#include "rvtx/system/molecule_ids.inl"

#endif // RVTX_SYSTEM_MOLECULE_IDS_HPP
