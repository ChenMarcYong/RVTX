#include "rvtx/dil/geometry/sphere_holder.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/molecule_ids.hpp"
#include "rvtx/dil/utils/buffer.hpp"
#include <rvtx/molecule/color.hpp> 
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace rvtx::dil
{

    struct Sphere
    {
        Diligent::float3 position;
        float radius;
        Diligent::float3 color;
        float visibility;
    };

    static rvtx::dil::SphereHolder buildSphereHolder(
        Diligent::IRenderDevice* device,
        const rvtx::Molecule& molecule,
        const rvtx::MoleculeIDs* moleculeIds,
        std::size_t                    startIndex,
        std::size_t                    endIndex)
    {
        using rvtx::dil::BufferBind;
        using rvtx::dil::BufferUsage;

        rvtx::dil::SphereHolder holder;
        holder.size = static_cast<uint32_t>(endIndex - startIndex);

        std::vector<Sphere>   spheres;
        spheres.reserve(holder.size);

        std::vector<uint32_t> ids;
        ids.reserve(holder.size);

        for (std::size_t i = startIndex; i < endIndex; ++i)
        {
            const auto& atom = molecule.atoms[i];
            const auto& pd = molecule.data[i];           // (x,y,z,r)
            const auto& residue = molecule.residues[atom.residueId];
            const auto& chain = molecule.chains[residue.chainId];

            const glm::vec3 col = getChainColor(chain);

            spheres.push_back({
                Diligent::float3{pd.x, pd.y, pd.z},                  // position locale
                pd.w,                                                // rayon
                Diligent::float3{col.x, col.y, col.z},               // couleur
                1.0f                                                 // visibilité
                });

            ids.push_back(moleculeIds ? (moleculeIds->atomIds.start + static_cast<uint32_t>(i)) : 0u);

        }

        holder.buffer = rvtx::dil::Buffer::Typed<Sphere>(
            device,
            rvtx::ConstSpan<Sphere>{spheres.data(), spheres.size()},
            BufferBind::ShaderResource,
            BufferUsage::Immutable,
            true
        );

        holder.idsBuffer = rvtx::dil::Buffer::Typed<uint32_t>(
            device,
            rvtx::ConstSpan<uint32_t>{ids.data(), ids.size()},
            BufferBind::ShaderResource,
            BufferUsage::Immutable,
            true
        );

        return holder;
    }

    SphereHolder SphereHolder::getMolecule(Diligent::IRenderDevice* device,
        const rvtx::Molecule& molecule,
        const rvtx::MoleculeIDs* moleculeIds)
    {
        return buildSphereHolder(device, molecule, moleculeIds,
            molecule.residentAtoms.start, molecule.residentAtoms.end);
    }

    SphereHolder SphereHolder::getNonResident(Diligent::IRenderDevice* device,
        const rvtx::Molecule& molecule,
        const rvtx::MoleculeIDs* moleculeIds)
    {
        return buildSphereHolder(device, molecule, moleculeIds,
            molecule.residentAtoms.start, molecule.residentAtoms.end);

    }

    SphereHolder SphereHolder::getSystem(Diligent::IRenderDevice* device,
        const rvtx::Molecule& molecule,
        const rvtx::MoleculeIDs* moleculeIds)
    {
        return buildSphereHolder(device, molecule, moleculeIds,
            0, molecule.atoms.size());
    }

} // namespace rvtx::dil
