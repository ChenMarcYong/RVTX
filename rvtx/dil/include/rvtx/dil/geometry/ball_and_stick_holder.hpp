#ifndef RVTX_DIL_BALL_AND_STICK_HOLDER_HPP
#define RVTX_DIL_BALL_AND_STICK_HOLDER_HPP

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
    struct BallAndStickHolder
    {
        BallAndStickHolder() = default;
        ~BallAndStickHolder() = default;

        BallAndStickHolder(const BallAndStickHolder&) = delete;
        BallAndStickHolder& operator=(const BallAndStickHolder&) = delete;
        BallAndStickHolder(BallAndStickHolder&&) noexcept = default;
        BallAndStickHolder& operator=(BallAndStickHolder&&) noexcept = default;

        static BallAndStickHolder getMolecule(Diligent::IRenderDevice* device,
            const rvtx::Molecule& molecule,
            const rvtx::MoleculeIDs* moleculeIds = nullptr);

        static BallAndStickHolder getNonResident(Diligent::IRenderDevice* device,
            const rvtx::Molecule& molecule,
            const rvtx::MoleculeIDs* moleculeIds = nullptr);

        static BallAndStickHolder getSystem(Diligent::IRenderDevice* device,
            const rvtx::Molecule& molecule,
            const rvtx::MoleculeIDs* moleculeIds = nullptr);

        uint32_t      size = 0;
        rvtx::dil::Buffer buffer;       // sphères
        rvtx::dil::Buffer idsBuffer;    // ids
        float         additionalRadius = 0.f;


        //uint32_t      sizeSphere = 0;
        //rvtx::dil::Buffer bufferSphere;       // sphères
        //rvtx::dil::Buffer idsBufferSphere;    // ids
        //float         additionalRadiusSphere = 0.f;

    };
}



#endif
