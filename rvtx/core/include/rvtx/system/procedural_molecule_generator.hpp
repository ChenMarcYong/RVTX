#ifndef RVTX_SYSTEM_PROCEDURAL_MOLECULE_GENERATOR_HPP
#define RVTX_SYSTEM_PROCEDURAL_MOLECULE_GENERATOR_HPP

#include <functional>
#include <string>
#include <vector>

#include "rvtx/molecule/molecule.hpp"

namespace rvtx
{
    /**
     * Class to create procedural molecules.
     *
     * The user should at least specify :
     *  - A molecule database using 'moleculeDB' or 'loadMoleculesDB' functions
     *  - A shape to create the procedural molecule with using the 'shape' function.
     *
     * The generation of the molecule uses an occupancy set so that atoms generated over other pre-existing atoms are
     * skipped. This grid can be of any size and can be set using the 'atomSpacing' function. The
     * 'rvtx::ProceduralMoleculeGenerator::Spacing' struct provides pre-defined settings. An higher spacing decreases
     * computation time but make atoms more sparse.
     *
     * If you wish to skip the placement of atoms (to create pockets), set the 'skipProbability' between ]0; 1].
     * You are also able to set different molecule for the skip process using 'skipMoleculeDB' or 'loadSkipMoleculesDB'
     * functions. If a 'skipMoleculeDB' has not been set, the skip process uses molecules from the 'moleculeDB'.
     *
     */
    class ProceduralMoleculeGenerator
    {
      public:
        // Defines the shape of the procedural molecule. 'containsFunc' needs to be set
        // Then either 'samplingSpace' or 'sampleFunc' should be defined
        struct Shape
        {
            Shape() = default;
            Shape( const AABB & samplingSpace, const std::function<bool( const glm::vec3 & )> & containsFunc );
            Shape( const AABB & samplingSpace, const std::function<glm::vec3()> & sampleFunc );

            Shape( const Shape & )             = default;
            Shape & operator=( const Shape & ) = default;
            Shape( Shape && ) noexcept;
            Shape & operator=( Shape && ) noexcept;

            std::function<bool( const glm::vec3 & )> containsFunc;

            // Either overload the 'sampleFunc', or define the sampling space bounding box (for linear rand)

            // The [min, max] values used for the random generator, [min, max] should approximate the shape AABB.
            AABB samplingSpace;
            // A custom sampling function for the shape
            std::function<glm::vec3()> sampleFunc {};

            // By default, if (sampleFunc == false), uses rejection sampling using 'containsFunc'
            glm::vec3 sample() const;
        };

        struct Spacing
        {                                             // Tested on 1AGA, 8F5T and 6L7E
            static constexpr float lossless = 0.76f;  // No atom loss
            static constexpr float minimal  = 1.35f;  // About 5% loss
            static constexpr float balanced = 1.63f;  // About 12.5% loss
            static constexpr float medium   = 2.075f; // About 25% loss
            static constexpr float high     = 3.0f;   // About 50% loss
        };

        ProceduralMoleculeGenerator()  = default;
        ~ProceduralMoleculeGenerator() = default;

        inline ProceduralMoleculeGenerator & shape( const Shape & sampledShape );
        inline ProceduralMoleculeGenerator & relaxedPlacement( const bool relaxedPlacement );
        inline ProceduralMoleculeGenerator & moleculeDB( const ConstSpan<rvtx::Molecule> molecules );
        inline ProceduralMoleculeGenerator & skipMoleculeDB( const ConstSpan<rvtx::Molecule> molecules );
        inline ProceduralMoleculeGenerator & samplesCount( const uint32_t count );
        inline ProceduralMoleculeGenerator & atomSpacing( const float spacing );
        inline ProceduralMoleculeGenerator & skipProbability( const float probability );
        inline ProceduralMoleculeGenerator & randomChainID( const bool      enabled,
                                                            RangeBase<char> colorRange = { 'A', 'Z' } );
        inline ProceduralMoleculeGenerator & randomChainID( const bool enabled, const ConstSpan<char> colorIDs );
        inline ProceduralMoleculeGenerator & minimumAtomPerPlacement( const std::size_t minimum );

        ProceduralMoleculeGenerator & loadMoleculesDB( const std::vector<std::string> & moleculesFileNames );
        ProceduralMoleculeGenerator & loadSkipMoleculesDB( const std::vector<std::string> & moleculesFileNames );

        Molecule generate() const;

      private:
        Shape                 m_sampledShape;
        std::vector<Molecule> m_moleculesDB;
        std::vector<Molecule> m_skipMoleculesDB;
        uint32_t              m_samplesCount { 100 };
        std::size_t           m_minimumAtomPerPlacement { 0 };
        bool                  m_relaxedPlacement { true };
        glm::vec3             m_atomSpacing { Spacing::balanced };
        float                 m_skipProbability { 0.f };
        bool                  m_randomChainColor { false };
        std::vector<char>     m_randomChainColorIDs { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };
    };
} // namespace rvtx

#include "procedural_molecule_generator.inl"

#endif // RVTX_SYSTEM_PROCEDURAL_MOLECULE_GENERATOR_HPP
