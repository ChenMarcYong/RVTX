#include "rvtx/system/procedural_molecule_generator.hpp"

#include "bindings/defines.hpp"
#include "pyrvtx/py_molecule.hpp"
#include "pyrvtx/py_scene.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( ProceduralMoleculeGenerator )
    {
        nb::class_<ProceduralMoleculeGenerator> pmg( m, "MoleculeGenerator" );

        nb::class_<ProceduralMoleculeGenerator::Shape>( pmg, "Shape" )
            .def( nb::init<>() )
            .def( nb::init<const AABB &, const std::function<bool( const glm::vec3 & )> &>(),
                  "aabb"_a,
                  "contains_function"_a )
            .def( nb::init<const AABB &, const std::function<glm::vec3()> &>(), "aabb"_a, "sample_function"_a )
            .def_rw( "contains_function", &ProceduralMoleculeGenerator::Shape::containsFunc )
            .def_rw( "sampling_space", &ProceduralMoleculeGenerator::Shape::samplingSpace )
            .def_rw( "sample_function", &ProceduralMoleculeGenerator::Shape::sampleFunc )
            .def( "sample", &ProceduralMoleculeGenerator::Shape::sample );

        nb::class_<ProceduralMoleculeGenerator::Spacing>( pmg, "Spacing" )
            .def_ro_static( "lossless", &ProceduralMoleculeGenerator::Spacing::lossless )
            .def_ro_static( "minimal", &ProceduralMoleculeGenerator::Spacing::minimal )
            .def_ro_static( "balanced", &ProceduralMoleculeGenerator::Spacing::balanced )
            .def_ro_static( "medium", &ProceduralMoleculeGenerator::Spacing::medium )
            .def_ro_static( "high", &ProceduralMoleculeGenerator::Spacing::high );

        pmg.def( nb::init<>() )
            .def( "shape", &ProceduralMoleculeGenerator::shape, "sampled_shape"_a, nb::rv_policy::reference )
            .def( "relaxed_placement",
                  &ProceduralMoleculeGenerator::relaxedPlacement,
                  "relaxed"_a,
                  nb::rv_policy::reference )
            .def( "molecule_db", &ProceduralMoleculeGenerator::moleculeDB, "molecules"_a, nb::rv_policy::reference )
            .def( "skip_molecule_db",
                  &ProceduralMoleculeGenerator::skipMoleculeDB,
                  "molecules"_a,
                  nb::rv_policy::reference )
            .def( "samples_count", &ProceduralMoleculeGenerator::samplesCount, "count"_a, nb::rv_policy::reference )
            .def( "atom_spacing", &ProceduralMoleculeGenerator::atomSpacing, "spacing"_a, nb::rv_policy::reference )
            .def( "skip_probability",
                  &ProceduralMoleculeGenerator::skipProbability,
                  "probability"_a,
                  nb::rv_policy::reference )
            .def(
                "random_chain_id",
                []( ProceduralMoleculeGenerator & pmg, const bool enabled, const char start, const char end ) {
                    return pmg.randomChainID( enabled, { start, end } );
                },
                "enabled"_a = true,
                "start"_a   = 'A',
                "end"_a     = "Z",
                nb::rv_policy::reference )
            .def(
                "random_chain_id",
                []( ProceduralMoleculeGenerator & pmg, const bool enabled, const std::vector<char> & colorIDs )
                { return pmg.randomChainID( enabled, colorIDs ); },
                "enabled"_a,
                "color_ids"_a,
                nb::rv_policy::reference )
            .def( "minimum_atom_per_placement",
                  &ProceduralMoleculeGenerator::minimumAtomPerPlacement,
                  "minimum"_a,
                  nb::rv_policy::reference )
            .def( "load_molecules_db",
                  &ProceduralMoleculeGenerator::loadMoleculesDB,
                  "molecules_file_names"_a,
                  nb::rv_policy::reference )
            .def( "load_skip_molecules_db",
                  &ProceduralMoleculeGenerator::loadSkipMoleculesDB,
                  "molecules_file_names"_a,
                  nb::rv_policy::reference );

        m.def( "create_molecule",
               &PyMolecule::createProcedural,
               "generator"_a,
               "scene"_a = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );
    }
} // namespace rvtx
