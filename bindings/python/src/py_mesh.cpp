#include "pyrvtx/py_mesh.hpp"

#include "pyrvtx/py_scene.hpp"

namespace rvtx
{
    PyMesh PyMesh::load( const std::filesystem::path & path, PyScene * scene )
    {
        PyMesh pyMesh = scene->loadMesh( path );

        pyMesh.scene = scene;

        return pyMesh;
    }

    PyMesh::~PyMesh()
    {
        if ( scene != nullptr && scene->registry.valid( self ) )
        {
            scene->registry.destroy( self );
        }
    }

    PyMesh::PyMesh( PyMesh && other ) noexcept
    {
        vertices  = std::exchange( other.vertices, nullptr );
        ids       = std::exchange( other.ids, nullptr );
        indices   = std::exchange( other.indices, nullptr );
        aabb      = std::exchange( other.aabb, nullptr );
        transform = std::exchange( other.transform, nullptr );
        visible   = std::exchange( other.visible, nullptr );
        self      = std::exchange( other.self, entt::handle {} );
        scene     = std::exchange( other.scene, nullptr );
    }

    PyMesh & PyMesh::operator=( PyMesh && other ) noexcept
    {
        std::swap( vertices, other.vertices );
        std::swap( ids, other.ids );
        std::swap( indices, other.indices );
        std::swap( aabb, other.aabb );
        std::swap( transform, other.transform );
        std::swap( visible, other.visible );
        std::swap( self, other.self );
        std::swap( scene, other.scene );

        return *this;
    }

    Mesh & PyMesh::getMesh() const { return self.get<Mesh>(); }
} // namespace rvtx
