#include "rvtx/system/procedural_molecule_generator.hpp"

namespace rvtx
{
    // Defines the shape of the procedural molecule.
    inline ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::shape( const Shape & sampledShape )
    {
        m_sampledShape = sampledShape;

        return *this;
    }

    // If enabled, only the position of the center of the molecule is checked to be in the shape
    // Disabled, all individual atoms will be checked.
    inline ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::relaxedPlacement( const bool relaxedPlacement )
    {
        m_relaxedPlacement = relaxedPlacement;

        return *this;
    }

    // Defines the probability for a molecule placement to be skipped. The atom will still contribute to the occupancy,
    // creating empty space where the molecule should have been.
    inline ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::skipProbability( const float probability )
    {
        m_skipProbability = probability;

        return *this;
    }

    // Sets if the chains should be randomly colored and with which color range.
    inline ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::randomChainID( const bool      enabled,
                                                                                     RangeBase<char> colorRange )
    {
        m_randomChainColor = enabled;

        m_randomChainColorIDs.clear();
        m_randomChainColorIDs.reserve( colorRange.size() );
        for ( char id = colorRange.start; id <= colorRange.end; id++ )
            m_randomChainColorIDs.emplace_back( id );

        return *this;
    }

    // Sets if the chains should be randomly colored and with which colors.
    inline ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::randomChainID( const bool            enabled,
                                                                                     const ConstSpan<char> colorIDs )
    {
        m_randomChainColor = enabled;
        m_randomChainColorIDs.clear();
        m_randomChainColorIDs.assign( colorIDs.ptr, colorIDs.ptr + colorIDs.size );

        return *this;
    }

    // Sets the minimum of atoms per placement of molecules, a higher number will remove the atoms but keep their
    // occupancy.
    inline ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::minimumAtomPerPlacement(
        const std::size_t minimum )
    {
        m_minimumAtomPerPlacement = minimum;

        return *this;
    }

    // Sets the molecule that should be sampled to create the procedural molecule.
    inline ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::moleculeDB( const ConstSpan<Molecule> molecules )
    {
        m_moleculesDB.assign( molecules.ptr, molecules.ptr + molecules.size );

        return *this;
    }

    // Sets the molecule that should be sampled to create the procedural molecule when a molecule is skipped
    inline ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::skipMoleculeDB(
        const ConstSpan<rvtx::Molecule> molecules )
    {
        m_skipMoleculesDB.assign( molecules.ptr, molecules.ptr + molecules.size );

        return *this;
    }

    // Sets the number of molecules that should be put inside the procedural molecule.
    inline ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::samplesCount( const uint32_t count )
    {
        m_samplesCount = count;

        return *this;
    }

    // Sets the spacing of the grid used to avoid overlap, higher value will make the molecule more sparse.
    // Use the 'rvtx::ProceduralMoleculeGenerator::Spacing' struct values for pre-defined spacing settings
    inline ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::atomSpacing( const float spacing )
    {
        m_atomSpacing = glm::vec3 { spacing };

        return *this;
    }
} // namespace rvtx