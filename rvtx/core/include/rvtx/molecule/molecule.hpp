#ifndef RVTX_MOLECULE_MOLECULE_HPP
#define RVTX_MOLECULE_MOLECULE_HPP

#include <string>
#include <vector>

#include <glm/vec4.hpp>
#include <rvtx/core/logger.hpp>
#include <rvtx/system/acceleration_structure.hpp>

#include "rvtx/core/aabb.hpp"
#include "rvtx/core/type.hpp"

namespace rvtx
{
    enum class ColorMode
    {
        Custom,
        Atom,
        Chain
    };

    enum class RepresentationType
    {
        vanDerWaals,
        BallAndStick,
        Ses,
        Sticks,
        Sas,
        Cartoon
    };

    enum class Symbol : uint8_t;
    struct Atom
    {
        Symbol      symbol;
        std::size_t residueId;

        [[nodiscard]] float            getRadius() const;
        [[nodiscard]] std::string_view getName() const;
    };

    struct Chain
    {
        std::string id;
        std::string name;
        Range       residues;
    };

    struct Residue
    {
        enum class Type
        {
            Molecule,
            Ligand,
            Ion,
            Water
        };

        std::size_t id;
        std::string name;
        Type        type;

        Range atoms;
        Range bonds;

        std::size_t chainId = 0;
    };

    struct Bond
    {
        std::size_t first;  // first atom id
        std::size_t second; // second atom id

        inline operator std::pair<std::size_t, std::size_t>() const;
    };

    struct Molecule
    {
        static constexpr auto in_place_delete = true;

        Molecule() = default;
        Molecule( std::string            id,
                  std::string            name,
                  std::vector<glm::vec4> data,
                  std::vector<Atom>      atoms,
                  std::vector<Bond>      bonds,
                  std::vector<Residue>   residues,
                  std::vector<Chain>     chains,
                  std::vector<float>     charge,
                  Range                  peptideBonds,
                  Range                  residentAtoms,
                  Aabb                   aabb = {} );

        ~Molecule() = default;

                   Molecule( const Molecule & )  = default;
        Molecule & operator=( const Molecule & ) = default;

                   Molecule( Molecule && other ) noexcept;
        Molecule & operator=( Molecule && other ) noexcept;

        std::string            id;
        std::string            name;
        std::vector<glm::vec4> data; // {.xyz: atom position, .w: atom radius}

        // Classification and relationships
        std::vector<Atom>    atoms;
        std::vector<Bond>    bonds;
        std::vector<Residue> residues;
        std::vector<Chain>   chains;
        std::vector<float>   charge;

        // Special cases
        Range peptideBonds {};  // residue-residue bonds indices (at the end of the buffer)
        Range residentAtoms {}; // all atoms which are part of the protein (in contrast to non-polymers) (at the start
                                // of the buffer)

        Aabb aabb;

        [[nodiscard]] Aabb getAabb() const;
        [[nodiscard]] Aabb getAABB() const;
        void               computeAabb();

        AccelerationStructure as;
        void                  buildAccelerationStructure( const std::size_t gridSize = 16 );

        float computeCharge( const glm::vec3 & position, float minDist ) const;
    };

    inline Symbol toSymbol( uint8_t atomicId );

    std::string_view getName( Symbol symbol );
    float            getRadius( Symbol symbol );
    bool             isIon( std::string_view name );
    bool             isH2O( std::string_view name );

} // namespace rvtx

#include "rvtx/molecule/molecule.inl"

#endif // RVTX_MOLECULE_MOLECULE_HPP
