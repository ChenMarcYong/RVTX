#ifndef RVTX_DIL_GEOMETRY_SPHERE_HOLDER_HPP
#define RVTX_DIL_GEOMETRY_SPHERE_HOLDER_HPP

#include <optional>
#include <vector>

#include "Common/interface/RefCntAutoPtr.hpp"
#include "Graphics/GraphicsEngine/interface/Buffer.h"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/molecule_ids.hpp"
#include "rvtx/dil/utils/program.hpp"
#include "rvtx/dil/geometry/handler.hpp"


#include "rvtx/dil/utils/buffer.hpp"

namespace rvtx::dil
{
    struct SphereHolder
    {
        SphereHolder() = default;
        ~SphereHolder() = default;

        SphereHolder(const SphereHolder&) = delete;
        SphereHolder& operator=(const SphereHolder&) = delete;
        SphereHolder(SphereHolder&&) noexcept = default;
        SphereHolder& operator=(SphereHolder&&) noexcept = default;

        static SphereHolder getMolecule(Diligent::IRenderDevice* device,
            const rvtx::Molecule& molecule,
            const rvtx::MoleculeIDs* moleculeIds = nullptr);

        static SphereHolder getNonResident(Diligent::IRenderDevice* device,
            const rvtx::Molecule& molecule,
            const rvtx::MoleculeIDs* moleculeIds = nullptr);

        static SphereHolder getSystem(Diligent::IRenderDevice* device,
            const rvtx::Molecule& molecule,
            const rvtx::MoleculeIDs* moleculeIds = nullptr);

        uint32_t      size = 0;
        rvtx::dil::Buffer buffer;       // sphères
        rvtx::dil::Buffer idsBuffer;    // ids
        float         additionalRadius = 0.f;
    };
}



#endif
