#include "rvtx/dil/geometry/sphere_holder.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/molecule_ids.hpp"
#include "rvtx/dil/utils/buffer.hpp"
#include <rvtx/molecule/color.hpp> // pour getChainColor si tu l’as
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
    
    static inline glm::vec3 srgbToLinear(glm::vec3 c)
    {
        auto f = [](float u) {
            return (u <= 0.04045f) ? (u / 12.92f)
                : std::pow((u + 0.055f) / 1.055f, 2.4f);
            };
        return { f(c.r), f(c.g), f(c.b) };
    }

    // Fonction interne factorisée pour éviter de répéter le code dans getMolecule, getNonResident, getSystem
    static rvtx::dil::SphereHolder2 buildSphereHolder(
        Diligent::IRenderDevice* device,
        const rvtx::Molecule& molecule,
        const rvtx::MoleculeIDs* moleculeIds,
        std::size_t                    startIndex,
        std::size_t                    endIndex)
    {
        using rvtx::dil::BufferBind;
        using rvtx::dil::BufferUsage;

        rvtx::dil::SphereHolder2 holder;
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

            // Couleur par chaîne (ou ta logique)
            const glm::vec3 col = getChainColor(chain);



            // Remplissage (positions en espace local de la molécule)
            spheres.push_back({
                Diligent::float3{pd.x, pd.y, pd.z},                  // position locale
                pd.w,                                                // rayon
                Diligent::float3{col.x, col.y, col.z},               // couleur
                1.0f                                                 // visibilité
                });

            ids.push_back(moleculeIds ? (moleculeIds->atomIds.start + static_cast<uint32_t>(i)) : 0u);

        }

        // Buffer structuré des sphères (SRV)
        holder.buffer = rvtx::dil::Buffer::Typed<Sphere>(
            device,
            rvtx::ConstSpan<Sphere>{spheres.data(), spheres.size()},
            BufferBind::ShaderResource,
            BufferUsage::Immutable,
            /*structured*/ true
        );

        // Buffer structuré des IDs (SRV)
        holder.idsBuffer = rvtx::dil::Buffer::Typed<uint32_t>(
            device,
            rvtx::ConstSpan<uint32_t>{ids.data(), ids.size()},
            BufferBind::ShaderResource,
            BufferUsage::Immutable,
            /*structured*/ true
        );

        return holder;
    }

    SphereHolder2 SphereHolder2::getMolecule(Diligent::IRenderDevice* device,
        const rvtx::Molecule& molecule,
        const rvtx::MoleculeIDs* moleculeIds)
    {
        return buildSphereHolder(device, molecule, moleculeIds,
            molecule.residentAtoms.start, molecule.residentAtoms.end);
    }

    SphereHolder2 SphereHolder2::getNonResident(Diligent::IRenderDevice* device,
        const rvtx::Molecule& molecule,
        const rvtx::MoleculeIDs* moleculeIds)
    {
        return buildSphereHolder(device, molecule, moleculeIds,
            molecule.residentAtoms.start, molecule.residentAtoms.end);

    }

    SphereHolder2 SphereHolder2::getSystem(Diligent::IRenderDevice* device,
        const rvtx::Molecule& molecule,
        const rvtx::MoleculeIDs* moleculeIds)
    {
        return buildSphereHolder(device, molecule, moleculeIds,
            0, molecule.atoms.size());
    }

} // namespace rvtx::dil
