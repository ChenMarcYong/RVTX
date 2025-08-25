#include "rvtx/molecule/loader.hpp"

#include <chemfiles.hpp>
#include <glm/vec3.hpp>

#include "rvtx/core/logger.hpp"

namespace rvtx
{
    static void prepareChemfiles()
    {
#ifndef NDEBUG
        chemfiles::warning_callback_t callback = []( const std::string & p_log ) { logger::warning( p_log ); };
#else
        chemfiles::warning_callback_t callback = []( const std::string & p_log ) { /* logger::warning( p_log ); */ };
#endif
        chemfiles::set_warning_callback( callback );
    }

    Molecule load( chemfiles::Trajectory & trajectory );

    Molecule load( const std::filesystem::path & path )
    {
        if ( !std::filesystem::exists( path ) )
        {
            logger::error( "Molecule file '{}' not found.", path.string() );
            throw std::runtime_error( fmt::format( "Molecule file '{}' not found.", path.string() ) );
        }

        prepareChemfiles();

        chemfiles::Trajectory trajectory { path.string() };
        return load( trajectory );
    }

    Molecule load( const std::string_view buffer, const std::string & extension )
    {
        prepareChemfiles();
        auto trajectory = chemfiles::Trajectory::memory_reader( buffer.data(), buffer.size(), extension );

        return load( trajectory );
    }

    Molecule load( chemfiles::Trajectory & trajectory )
    {
        logger::debug( "{} frames found", trajectory.nsteps() );

        if ( trajectory.nsteps() == 0 )
            throw std::runtime_error( "Trajectory is empty" );

        chemfiles::Frame frame = trajectory.read();

        const std::string                       id        = frame.get( "pdb_idcode" ).value_or( "" ).as_string();
        const chemfiles::Topology &             topology  = frame.topology();
        const std::vector<chemfiles::Residue> & cResidues = topology.residues();
        const std::vector<chemfiles::Bond> &    cBonds    = topology.bonds();

        if ( frame.size() != topology.size() )
            throw std::runtime_error( "Data count mismatch" );

        const std::string name = frame.get( "name" ).value_or( "Unknown" ).as_string();

        // If no residue, create a fake one.
        if ( cResidues.empty() )
        {
            logger::warning( "No residues found" );
            chemfiles::Residue residue = chemfiles::Residue( "" );
            for ( std::size_t i = 0; i < frame.size(); ++i )
                residue.add_atom( i );
            frame.add_residue( residue );
        }

        std::vector<Chain>     chains {};
        std::vector<float>     charge {};
        std::vector<Atom>      shuffledAtoms {};
        std::vector<glm::vec4> shuffledData {};

        shuffledAtoms.reserve( frame.size() );
        shuffledData.reserve( frame.size() );

        std::unordered_map<std::size_t, std::vector<Bond>> mapResidueBonds {};
        mapResidueBonds.reserve( cResidues.size() );

#ifndef NDEBUG
        std::size_t residuesWithOneAtomCount = 0;
#endif

        std::map<std::size_t, Residue> residues {};
        std::map<std::size_t, Residue> nonPolymers {};
        for ( std::size_t residueIdx = 0; residueIdx < cResidues.size(); ++residueIdx )
        {
            const chemfiles::Residue & cResidue = cResidues[ residueIdx ];

            // Check if chain name changed.
            const std::string chainName = cResidue.properties().get( "chainname" ).value_or( "" ).as_string();
            const std::string chainId   = cResidue.properties().get( "chainid" ).value_or( "" ).as_string();

            if ( chains.empty() || chainName != chains.back().name )
                chains.emplace_back( Chain { chainId, chainName, Range { residueIdx, residueIdx } } );

            Chain & currentChain = chains.back();
            currentChain.residues.end++;

            mapResidueBonds.emplace( residueIdx, std::vector<Bond> {} );
            if ( cResidue.size() == 0 )
                logger::warning( "Empty residue found" );
            mapResidueBonds[ residueIdx ].reserve( cResidue.size() );

            const auto compositionType = cResidue.get( "composition_type" );
            const bool isMonomer       = cResidue.name() != "UNK"
                                   && ( !compositionType || ( compositionType && *compositionType != "NON-POLYMER" ) );

            Residue::Type type = Residue::Type::Molecule;
            if ( isIon( cResidue.name() ) )
                type = Residue::Type::Ion;
            else if ( isH2O( cResidue.name() ) )
                type = Residue::Type::Water;
            else if ( !isMonomer && cResidue.size() == 1 && *topology[ *cResidue.begin() ].atomic_number() == 8 )
                type = Residue::Type::Water;
            else if ( !isMonomer )
                type = Residue::Type::Ligand;

            // clang-format off
            Residue residue {
                residueIdx,
                cResidue.name(),
                type,
                Range { shuffledAtoms.size(), shuffledAtoms.size() + cResidue.size() },
                Range { 0, 0 }, chains.size() - 1
            };
            // clang-format on

#ifndef NDEBUG
            if ( cResidue.size() == 1 && type == Residue::Type::Molecule )
                residuesWithOneAtomCount++;
#endif

            if ( type == Residue::Type::Molecule )
                residues[ residueIdx ] = std::move( residue );
            else
                nonPolymers[ residueIdx ] = std::move( residue );

            for ( const std::size_t atomId : cResidue )
            {
                const chemfiles::Atom & cAtom = topology[ atomId ];

                chemfiles::span<chemfiles::Vector3D> positions = frame.positions();
                const chemfiles::Vector3D &          position  = positions[ atomId ];

                const Symbol symbol = toSymbol( cAtom.atomic_number().value_or( 0 ) );
                const Atom   atom { symbol, residueIdx };

                charge.emplace_back( static_cast<float>( cAtom.charge() ) );

                shuffledData.emplace_back( position[ 0 ], position[ 1 ], position[ 2 ], atom.getRadius() );
                shuffledAtoms.emplace_back( atom );
            }
        }

#ifndef NDEBUG
        logger::warning( "Detected {} residue{} of type molecule containing 1 atom",
                         residuesWithOneAtomCount,
                         residuesWithOneAtomCount > 1 ? "s" : "" );
#endif

        std::vector<Residue>   allResidues;
        std::vector<Atom>      atoms {};
        std::vector<glm::vec4> data {};

        std::unordered_map<std::size_t, std::size_t> topologyToAtomIdMap;
        topologyToAtomIdMap.reserve( shuffledData.size() );

        allResidues.reserve( cResidues.size() );
        atoms.reserve( frame.size() );
        data.reserve( frame.size() );
        for ( auto & [ _, residue ] : residues )
        {
            residue.id = allResidues.size();

            for ( std::size_t i = residue.atoms.start; i < residue.atoms.end; i++ )
            {
                Atom & atom    = shuffledAtoms[ i ];
                atom.residueId = residue.id;

                topologyToAtomIdMap[ i ] = atoms.size();

                atoms.emplace_back( atom );
                data.emplace_back( shuffledData[ i ] );
            }

            allResidues.emplace_back( std::move( residue ) );
        }

        Range residentAtoms { 0, atoms.size() };

        for ( auto & [ _, residue ] : nonPolymers )
        {
            residue.id = allResidues.size();

            for ( std::size_t i = residue.atoms.start; i < residue.atoms.end; i++ )
            {
                Atom & atom    = shuffledAtoms[ i ];
                atom.residueId = residue.id;

                topologyToAtomIdMap[ i ] = atoms.size();

                atoms.emplace_back( atom );
                data.emplace_back( shuffledData[ i ] );
            }

            allResidues.emplace_back( std::move( residue ) );
        }

        // Sort bonds by residues
        std::vector<Bond> peptideBonds {};
        for ( const auto & cBond : cBonds )
        {
            const std::size_t startAtomId = cBond[ 0 ];
            const std::size_t endAtomId   = cBond[ 1 ];

            const std::size_t startResidueId = shuffledAtoms[ startAtomId ].residueId;
            const std::size_t endResidueId   = shuffledAtoms[ endAtomId ].residueId;

            const Bond bond { topologyToAtomIdMap.at( cBond[ 0 ] ), topologyToAtomIdMap.at( cBond[ 1 ] ) };
            if ( startResidueId == endResidueId )
                mapResidueBonds[ startResidueId ].emplace_back( bond );
            else
                peptideBonds.emplace_back( bond );
        }

        std::vector<Bond> bonds {};
        bonds.resize( cBonds.size() );

        std::size_t offset = 0;
        for ( const auto & [ residueId, residueBonds ] : mapResidueBonds )
        {
            if ( residueBonds.empty() )
                continue;

            Residue * residue;
            if ( residues.find( residueId ) != residues.end() )
                residue = &residues[ residueId ];
            else
                residue = &nonPolymers[ residueId ];

            residue->bonds.start = offset;
            residue->bonds.end   = residue->bonds.start + residueBonds.size();

            std::memcpy( bonds.data() + offset, residueBonds.data(), residueBonds.size() * sizeof( Bond ) );

            offset += residueBonds.size();
        }

        // Bonds between residues.
        Range peptides { offset, offset + peptideBonds.size() };
        std::memcpy( bonds.data() + offset, peptideBonds.data(), peptideBonds.size() * sizeof( Bond ) );

        Molecule molecule = Molecule { id, name, data, atoms, bonds, allResidues, chains, charge, peptides, residentAtoms, {} };
        molecule.computeAabb();

        return molecule;
    }
} // namespace rvtx
