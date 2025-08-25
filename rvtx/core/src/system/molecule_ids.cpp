#include "rvtx/system/molecule_ids.hpp"

#include <atomic>

#include "rvtx/molecule/molecule.hpp"

namespace rvtx
{
    MoleculeIDs::MoleculeIDs( Molecule & moleculeRef ) : molecule( &moleculeRef )
    {
        static std::atomic<uint32_t> nextId = 1;

        if ( !molecule->atoms.empty() )
        {
            atomIds = Range { nextId, nextId + molecule->atoms.size() - 1 };
            nextId += molecule->atoms.size();
        }

        if ( !molecule->bonds.empty() )
        {
            bondIds = Range { nextId, nextId + molecule->bonds.size() - 1 };
            nextId += molecule->bonds.size();
        }
    }
} // namespace rvtx
