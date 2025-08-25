#include "rvtx/system/procedural_molecule_generator.hpp"

#include <utility>
#include <unordered_set>

#include "glm/gtc/constants.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtc/random.hpp"
#include "glm/gtx/hash.hpp"
#include "plf_nanotimer.h"
#include "rvtx/core/logger.hpp"
#include "rvtx/molecule/loader.hpp"

namespace rvtx
{
    ProceduralMoleculeGenerator::Shape::Shape( const AABB &                                     samplingSpace,
                                               const std::function<bool( const glm::vec3 & )> & containsFunc ) :
        samplingSpace( samplingSpace ),
        containsFunc( containsFunc )
    {
    }

    ProceduralMoleculeGenerator::Shape::Shape( const AABB &                       samplingSpace,
                                               const std::function<glm::vec3()> & sampleFunc ) :
        samplingSpace( samplingSpace ),
        sampleFunc( sampleFunc )
    {
    }

    ProceduralMoleculeGenerator::Shape::Shape( Shape && other ) noexcept
    {
        samplingSpace = std::exchange( other.samplingSpace, {} );
        containsFunc  = std::exchange( other.containsFunc, nullptr );
        sampleFunc    = std::exchange( other.sampleFunc, nullptr );
    }

    ProceduralMoleculeGenerator::Shape & ProceduralMoleculeGenerator::Shape::operator=( Shape && other ) noexcept
    {
        std::swap( samplingSpace, other.samplingSpace );
        std::swap( containsFunc, other.containsFunc );
        std::swap( sampleFunc, other.sampleFunc );

        return *this;
    }

    glm::vec3 ProceduralMoleculeGenerator::Shape::sample() const
    {
        if ( sampleFunc )
            return sampleFunc();

        glm::vec3 point;
        do
        {
            point = glm::linearRand( samplingSpace.min, samplingSpace.max );
        } while ( !containsFunc( point ) );

        return point;
    }

    // Loads molecule for the sampling of the procedural molecule from the disk.
    ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::loadMoleculesDB(
        const std::vector<std::string> & moleculesFileNames )
    {
        m_moleculesDB.reserve( moleculesFileNames.size() );
        for ( const std::string & moleculeFileName : moleculesFileNames )
            m_moleculesDB.emplace_back( rvtx::load( moleculeFileName ) );

        return *this;
    }

    // Loads molecule for the sampling of the procedural molecule when a molecule is skipped from the disk.
    ProceduralMoleculeGenerator & ProceduralMoleculeGenerator::loadSkipMoleculesDB(
        const std::vector<std::string> & moleculesFileNames )
    {
        m_skipMoleculesDB.reserve( moleculesFileNames.size() );
        for ( const std::string & moleculeFileName : moleculesFileNames )
            m_skipMoleculesDB.emplace_back( rvtx::load( moleculeFileName ) );

        return *this;
    }

    Molecule ProceduralMoleculeGenerator::generate() const
    {
        Molecule proceduralMolecule;

        plf::nanotimer timer;
        timer.start();

        std::unordered_set<glm::uvec3> occupancySet;
        for ( uint32_t i = 0; i < m_samplesCount; i++ )
        {
            // Probability to skip the molecule atom placement, to create pockets
            const bool skipMol = glm::linearRand( 0.f, 1.f ) < m_skipProbability;

            const glm::vec4 position { m_sampledShape.sample(), 0.f };
            const glm::quat rotationQuat { glm::linearRand( glm::vec3 { 0.f }, glm::vec3 { glm::two_pi<float>() } ) };
            const glm::mat4 rotation = mat4_cast( rotationQuat );

            std::size_t molRandIndex = glm::linearRand<std::size_t>(
                0, ( skipMol && !m_skipMoleculesDB.empty() ? m_skipMoleculesDB : m_moleculesDB ).size() - 1 );

            const Molecule & molecule = skipMol && !m_skipMoleculesDB.empty() ? m_skipMoleculesDB[ molRandIndex ]
                                                                              : m_moleculesDB[ molRandIndex ];

            // Add chains
            const std::size_t firstChainID = proceduralMolecule.chains.size();
            if ( m_randomChainColor )
            {
                for ( Chain chain : molecule.chains )
                {
                    chain.id
                        = m_randomChainColorIDs[ glm::linearRand<std::size_t>( 0, m_randomChainColorIDs.size() - 1 ) ];
                    proceduralMolecule.chains.emplace_back( chain );
                }
            }
            else
            {
                proceduralMolecule.chains.insert( proceduralMolecule.chains.begin(),
                                                  molecule.chains.begin(),
                                                  molecule.chains.end() ); // No data to transform
            }

            // Add residues
            const std::size_t firstResidueID = proceduralMolecule.residues.size();
            for ( auto residue : molecule.residues )
            {
                residue.chainId += firstChainID;
                proceduralMolecule.residues.emplace_back( residue );
            }

            // Add atoms
            std::vector<std::pair<Atom, glm::vec4>> atoms;
            atoms.reserve( molecule.residentAtoms.size() );
            for ( std::size_t j = 0; j < molecule.residentAtoms.size(); j++ )
            {
                glm::vec4 data = molecule.data[ j ];
                data           = position + rotation * data;
                data.w         = molecule.data[ j ].w;

                const glm::uvec3 gridPos = glm::vec3 { data } / m_atomSpacing;
                if ( occupancySet.find( gridPos ) == occupancySet.end() )
                {
                    if ( !m_relaxedPlacement && !m_sampledShape.containsFunc( glm::vec3 { data } ) )
                        continue;

                    occupancySet.emplace( gridPos );

                    if ( !skipMol )
                    {
                        Atom atom = molecule.atoms[ j ];
                        atom.residueId += firstResidueID;
                        atoms.emplace_back( atom, data );
                    }
                }
            }

            if ( !skipMol && atoms.size() > m_minimumAtomPerPlacement )
            {
                for ( const auto & [ atom, data ] : atoms )
                {
                    proceduralMolecule.data.emplace_back( data );
                    proceduralMolecule.atoms.emplace_back( atom );
                }
            }
        }

        proceduralMolecule.residentAtoms = Range { 0, proceduralMolecule.data.size() };

        logger::info( "Procedural molecule generated in {:.1f} ms, atom count : {}",
                      timer.get_elapsed_ms(),
                      proceduralMolecule.atoms.size() );

        proceduralMolecule.computeAabb();

        return proceduralMolecule;
    }
} // namespace rvtx
