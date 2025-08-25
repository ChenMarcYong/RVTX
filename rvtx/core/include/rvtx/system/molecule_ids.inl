#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/molecule_ids.hpp"

namespace rvtx
{
    inline bool inRange( const Range & range, const uint32_t element )
    {
        return range.start <= element && range.end >= element;
    }

    inline bool MoleculeIDs::contains( const uint32_t id ) const
    {
        return inRange( atomIds, id ) || inRange( bondIds, id );
    }

    inline std::size_t MoleculeIDs::atomIndex( const uint32_t id ) const
    {
        assert( contains( id ) && "This molecule does not contain this id" );
        return id - atomIds.start;
    }

    inline std::size_t MoleculeIDs::bondIndex( const uint32_t id ) const
    {
        assert( contains( id ) && "This molecule does not contain this id" );
        return id - bondIds.start;
    }

    inline Atom MoleculeIDs::getAtom( const uint32_t id ) const { return molecule->atoms[ atomIndex( id ) ]; }

    inline glm::vec4 MoleculeIDs::getData( const uint32_t id ) const { return molecule->data[ atomIndex( id ) ]; }

    inline Bond MoleculeIDs::getBond( const uint32_t id ) const { return molecule->bonds[ bondIndex( id ) ]; }

    inline std::array<Atom, 2> MoleculeIDs::getBondAtoms( const uint32_t id ) const
    {
        const Bond bond = getBond( id );

        return { molecule->atoms[ bond.first ], molecule->atoms[ bond.second ] };
    }

    inline std::array<glm::vec4, 2> MoleculeIDs::getBondData( const uint32_t id ) const
    {
        const Bond bond = getBond( id );

        return { molecule->data[ bond.first ], molecule->data[ bond.second ] };
    }
} // namespace rvtx
