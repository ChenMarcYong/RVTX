#ifndef RVTX_MESH_LOADER_HPP
#define RVTX_MESH_LOADER_HPP

#include "rvtx/core/filesystem.hpp"
#include "rvtx/mesh/mesh.hpp"

namespace rvtx
{
    Mesh loadMesh( const std::filesystem::path & path );
} // namespace rvtx

#endif // RVTX_MESH_LOADER_HPP
