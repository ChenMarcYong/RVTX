#ifndef PYRVTX_PY_MOLECULE_HPP
#define PYRVTX_PY_MOLECULE_HPP

#include <filesystem>

#include <entt/entt.hpp>
#include <rvtx/molecule/molecule.hpp>
#include <rvtx/system/molecule_ids.hpp>
#include <rvtx/system/procedural_molecule_generator.hpp>

namespace rvtx
{
    struct PyScene;
    struct Transform;

    struct PyRepresentation
    {
        static constexpr bool in_place_delete = true;

        PyRepresentation( const RepresentationType representation );

        RepresentationType representation = RepresentationType::vanDerWaals;
    };

    struct PyMolecule
    {
        static PyMolecule load( const std::filesystem::path & path,
                                const RepresentationType      representationType,
                                PyScene *                     scene );

        static PyMolecule createProcedural( const ProceduralMoleculeGenerator & generator, PyScene * scene );

        PyMolecule() = default;
        ~PyMolecule();

        PyMolecule( const PyMolecule & )             = delete;
        PyMolecule & operator=( const PyMolecule & ) = delete;

        PyMolecule( PyMolecule && other ) noexcept;
        PyMolecule & operator=( PyMolecule && other ) noexcept;

        void setRepresentation( const RepresentationType newRepresentation ) const;

        std::string *            id;
        std::string *            name;
        std::vector<glm::vec4> * data;

        std::vector<Atom> *    atoms;
        std::vector<Bond> *    bonds;
        std::vector<Residue> * residues;
        std::vector<Chain> *   chains;

        Range * peptideBonds;
        Range * residentAtoms;

        PyRepresentation * representation;

        Aabb *        aabb;
        MoleculeIDs * ids;

        Transform * transform;
        bool *      visible;

        entt::handle self;
        PyScene *    scene;
    };
} // namespace rvtx

#endif