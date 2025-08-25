#include "pyrvtx/py_point_cloud.hpp"

#include "pyrvtx/py_scene.hpp"

namespace rvtx
{
    PyPointCloud::PyPointCloud( PyPointCloud && other ) noexcept
    {
        points       = std::exchange( other.points, {} );
        pointsColors = std::exchange( other.pointsColors, {} );
        needsUpdate  = std::exchange( other.needsUpdate, false );
#if RVTX_GL
        holder = std::exchange( other.holder, nullptr );
#endif
    }

    PyPointCloud & PyPointCloud::operator=( PyPointCloud && other ) noexcept
    {
        std::swap( points, other.points );
        std::swap( pointsColors, other.pointsColors );
        std::swap( needsUpdate, other.needsUpdate );
#if RVTX_GL
        std::swap( holder, other.holder );
#endif

        return *this;
    }

    void PyPointCloud::update()
    {
#if RVTX_GL
        if ( holder == nullptr )
        {
            needsUpdate = true;
            return;
        }

        holder->nodesCount = static_cast<uint32_t>( points.size() );
        if ( holder->nodesCount > 0 )
        {
            holder->nodesBuffer       = gl::Buffer::Typed<glm::vec4>( points );
            holder->nodesColorsBuffer = gl::Buffer::Typed<glm::vec4>( pointsColors );
        }

        needsUpdate = false;
#endif
    }

    PyPointCloudView::~PyPointCloudView()
    {
        if ( scene != nullptr && scene->registry.valid( self ) )
        {
            scene->registry.destroy( self );
        }
    }

    PyPointCloudView::PyPointCloudView( PyPointCloudView && other ) noexcept
    {
        points       = std::exchange( other.points, {} );
        pointsColors = std::exchange( other.pointsColors, {} );
        pointCloud   = std::exchange( other.pointCloud, nullptr );
        scene        = std::exchange( other.scene, nullptr );
        self         = std::exchange( other.self, entt::handle {} );
    }

    PyPointCloudView & PyPointCloudView::operator=( PyPointCloudView && other ) noexcept
    {
        std::swap( points, other.points );
        std::swap( pointsColors, other.pointsColors );
        std::swap( pointCloud, other.pointCloud );
        std::swap( scene, other.scene );
        std::swap( self, other.self );

        return *this;
    }

    void PyPointCloudView::update() { pointCloud->update(); }

    PyPointCloudView PyPointCloudView::createPointCloud( const std::vector<glm::vec3> & points,
                                                         const std::vector<glm::vec3> & colors,
                                                         const std::vector<float> &     radii,
                                                         PyScene &                      scene )
    {
        PyPointCloudView view = scene.createPointCloud( points, colors, radii );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.pointCloud->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.pointCloud->update();
        }
#endif

        return view;
    }

    PyPointCloudView PyPointCloudView::createPointCloud( const std::vector<glm::vec3> & points,
                                                         const glm::vec3 &              color,
                                                         const float                    radius,
                                                         PyScene &                      scene )
    {
        PyPointCloudView view = scene.createPointCloud( points, color, radius );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.pointCloud->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.pointCloud->update();
        }
#endif

        return view;
    }

    PyPointCloudView PyPointCloudView::createPointCloud( const std::vector<glm::vec3> & points,
                                                         const std::vector<float> &     radii,
                                                         const std::vector<glm::vec3> & colors,
                                                         PyScene &                      scene )
    {
        PyPointCloudView view = scene.createPointCloud( points, radii, colors );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.pointCloud->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.pointCloud->update();
        }
#endif

        return view;
    }

    PyPointCloudView PyPointCloudView::createPointCloud( const std::vector<glm::vec3> & points,
                                                         const float                    radius,
                                                         const glm::vec3 &              color,
                                                         PyScene &                      scene )
    {
        PyPointCloudView view = scene.createPointCloud( points, radius, color );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.pointCloud->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.pointCloud->update();
        }
#endif

        return view;
    }

    PyPointCloudView PyPointCloudView::createPointCloud( const std::vector<glm::vec4> & points,
                                                         const std::vector<glm::vec4> & colors,
                                                         PyScene &                      scene )
    {
        PyPointCloudView view = scene.createPointCloud( points, colors );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.pointCloud->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.pointCloud->update();
        }
#endif

        return view;
    }

    PyPointCloudView PyPointCloudView::createPointCloud( const std::vector<glm::vec4> & points,
                                                         const glm::vec4 &              color,
                                                         PyScene &                      scene )
    {
        PyPointCloudView view = scene.createPointCloud( points, color );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.pointCloud->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.pointCloud->update();
        }
#endif

        return view;
    }

    PyPointCloudView PyPointCloudView::createPointCloud( const std::vector<glm::vec4> & points,
                                                         const std::vector<glm::vec3> & colors,
                                                         PyScene &                      scene )
    {
        PyPointCloudView view = scene.createPointCloud( points, colors );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.pointCloud->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.pointCloud->update();
        }
#endif

        return view;
    }
    PyPointCloudView PyPointCloudView::createPointCloud( const std::vector<glm::vec4> & points,
                                                         const glm::vec3 &              color,
                                                         PyScene &                      scene )
    {
        PyPointCloudView view = scene.createPointCloud( points, color );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.pointCloud->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.pointCloud->update();
        }
#endif

        return view;
    }
} // namespace rvtx
