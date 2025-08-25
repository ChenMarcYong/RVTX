#ifndef PYRVTX_PY_MESH_HPP
#define PYRVTX_PY_MESH_HPP

#include <filesystem>

#include <entt/entt.hpp>
#include <rvtx/mesh/mesh.hpp>

namespace rvtx
{
    struct PyScene;
    struct Transform;

    struct PyMesh
    {
        static PyMesh load( const std::filesystem::path & path, PyScene * scene );

         PyMesh() = default;
        ~PyMesh();

                 PyMesh( const PyMesh & )    = delete;
        PyMesh & operator=( const PyMesh & ) = delete;

                 PyMesh( PyMesh && other ) noexcept;
        PyMesh & operator=( PyMesh && other ) noexcept;

        std::vector<Mesh::Vertex> * vertices;
        std::vector<uint32_t> *     indices;
        std::vector<uint32_t> *     ids;

        Aabb * aabb;

        Transform * transform;
        bool *      visible;

        entt::handle self;
        PyScene *    scene;

        Mesh & getMesh() const;
    };
} // namespace rvtx

#endif