#ifndef PYRVTX_PYSCENE_HPP
#define PYRVTX_PYSCENE_HPP

#include <filesystem>

#include <rvtx/system/scene.hpp>

#include "pyrvtx/py_mesh.hpp"
#include "pyrvtx/py_camera.hpp"
#include "pyrvtx/py_graph.hpp"
#include "pyrvtx/py_molecule.hpp"
#include "pyrvtx/py_point_cloud.hpp"
#include "rvtx/mesh/mesh.hpp"

namespace rvtx
{
    struct PyEngine;

    struct PyScene : public Scene
    {
        PyScene() = default;

#ifdef RVTX_GL
        bool       hasPyEngine() const;
        PyEngine * getPyEngine() const;
#endif

        PyMolecule loadMolecule( const std::filesystem::path & path, const RepresentationType representationType );

        PyMesh loadMesh( const std::filesystem::path & path );

        PyMolecule createMolecule( const Molecule molecule, const RepresentationType representationType );

        PyMolecule createProceduralMolecule( const ProceduralMoleculeGenerator & generator );

        PyCamera createCamera( const Transform &            transform,
                               const Camera::Target         target,
                               const Camera::Projection     projectionType,
                               const CameraController::Type controller,
                               const glm::uvec2             viewport );

        PyPointCloudView createPointCloud( const std::vector<glm::vec3> & points,
                                           const std::vector<glm::vec3> & colors,
                                           const std::vector<float> &     radii );

        PyPointCloudView createPointCloud( const std::vector<glm::vec3> & points,
                                           const glm::vec3 &              colors,
                                           const float                    radii );

        PyPointCloudView createPointCloud( const std::vector<glm::vec3> & points,
                                           const std::vector<float> &     radii,
                                           const std::vector<glm::vec3> & colors );

        PyPointCloudView createPointCloud( const std::vector<glm::vec3> & points,
                                           const float                    radius,
                                           const glm::vec3 &              color );

        PyPointCloudView createPointCloud( const std::vector<glm::vec4> & points,
                                           const std::vector<glm::vec4> & colors );

        PyPointCloudView createPointCloud( const std::vector<glm::vec4> & points,
                                           const std::vector<glm::vec3> & colors );

        PyPointCloudView createPointCloud( const std::vector<glm::vec4> & points, const glm::vec4 & color );

        PyPointCloudView createPointCloud( const std::vector<glm::vec4> & points, const glm::vec3 & color );

        PyGraphView createGraph( const std::vector<glm::vec4> & nodes,
                                 const std::vector<unsigned> &  edges,
                                 const std::vector<glm::vec4> & nodesColors,
                                 const std::vector<glm::vec4> & edgesData );

        PyGraphView createGraph( const std::vector<glm::vec3> & nodes,
                                 const std::vector<unsigned> &  edges,
                                 const std::vector<float> &     nodesRadii,
                                 const std::vector<float> &     edgesRadii,
                                 const std::vector<glm::vec3> & nodesColors,
                                 const std::vector<glm::vec3> & edgesColors );

        PyGraphView createGraph( const std::vector<glm::vec3> & nodes,
                                 const std::vector<unsigned> &  edges,
                                 const float                    nodesRadius,
                                 const float                    edgesRadius,
                                 const glm::vec3 &              nodesColor,
                                 const glm::vec3 &              edgesColor );

        PyGraphView createGraph( const std::vector<glm::vec3> & nodes,
                                 const std::vector<unsigned> &  edges,
                                 const float                    edgesRadius,
                                 const glm::vec3 &              edgesColor );

        PyGraphView createGraph( const std::vector<glm::vec4> & nodes,
                                 const float                    edgesRadius,
                                 const glm::vec3 &              edgesColor,
                                 const PyGraph::ConnectionType  connectionType );

        PyGraphView createGraph( const std::vector<glm::vec3> & nodes,
                                 const float                    edgesRadius,
                                 const glm::vec3 &              edgesColor,
                                 const PyGraph::ConnectionType  connectionType );

        PyGraphView createGraph( const std::vector<glm::vec4> & nodes,
                                 const glm::vec3 &              edgesColor,
                                 const float                    edgesRadius,
                                 const PyGraph::ConnectionType  connectionType );

        PyGraphView createGraph( const std::vector<glm::vec3> & nodes,
                                 const glm::vec3 &              edgesColor,
                                 const float                    edgesRadius,
                                 const PyGraph::ConnectionType  connectionType );

        PyGraphView createGraph( const rvtx::Mesh & mesh,
                                 const glm::vec3 &  edgesColor,
                                 const float        edgesRadius,
                                 const glm::mat4 &  transform = glm::mat4 { 1.f } );

        PyGraphView createGraph( const Path<glm::vec3> & path,
                                 const uint32_t          numPoints,
                                 const glm::vec3 &       edgesColor,
                                 const float             edgesRadius,
                                 bool                    showKey,
                                 const glm::vec3 &       keyColors,
                                 const float             keyRadius );

        PyCamera * mainCamera;
    };
} // namespace rvtx

#endif