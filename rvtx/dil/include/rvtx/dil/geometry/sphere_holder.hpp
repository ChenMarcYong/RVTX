#ifndef RVTX_DIL_GEOMETRY_SPHERE_HOLDER_2_HPP
#define RVTX_DIL_GEOMETRY_SPHERE_HOLDER_2_HPP

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
    struct SphereHolder2
    {
        SphereHolder2() = default;
        ~SphereHolder2() = default;

        SphereHolder2(const SphereHolder2&) = delete;
        SphereHolder2& operator=(const SphereHolder2&) = delete;
        SphereHolder2(SphereHolder2&&) noexcept = default;
        SphereHolder2& operator=(SphereHolder2&&) noexcept = default;

        static SphereHolder2 getMolecule(Diligent::IRenderDevice* device,
            const rvtx::Molecule& molecule,
            const rvtx::MoleculeIDs* moleculeIds = nullptr);

        static SphereHolder2 getNonResident(Diligent::IRenderDevice* device,
            const rvtx::Molecule& molecule,
            const rvtx::MoleculeIDs* moleculeIds = nullptr);

        static SphereHolder2 getSystem(Diligent::IRenderDevice* device,
            const rvtx::Molecule& molecule,
            const rvtx::MoleculeIDs* moleculeIds = nullptr);

        uint32_t      size = 0;
        rvtx::dil::Buffer buffer;       // sphères
        rvtx::dil::Buffer idsBuffer;    // ids
        float         additionalRadius = 0.f;
    };
}



#endif
