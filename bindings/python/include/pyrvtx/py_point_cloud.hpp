#ifndef PYRVTX_PY_POINT_CLOUD_HPP
#define PYRVTX_PY_POINT_CLOUD_HPP

#if RVTX_GL
#include "rvtx/gl/system/debug_primitives.hpp"
#endif

namespace rvtx
{
    class PyScene;
    struct PyPointCloud;

    struct PyPointCloudView
    {
        PyPointCloudView() = default;
        ~PyPointCloudView();

        PyPointCloudView( const PyPointCloudView & )             = delete;
        PyPointCloudView & operator=( const PyPointCloudView & ) = delete;

        PyPointCloudView( PyPointCloudView && other ) noexcept;
        PyPointCloudView & operator=( PyPointCloudView && other ) noexcept;

        entt::handle self;
        PyScene *    scene { nullptr };

        std::vector<glm::vec4> * points;
        std::vector<glm::vec4> * pointsColors;

        PyPointCloud * pointCloud { nullptr };

        void update();

        static PyPointCloudView createPointCloud( const std::vector<glm::vec3> & points,
                                                  const std::vector<glm::vec3> & colors,
                                                  const std::vector<float> &     radii,
                                                  PyScene &                      scene );

        static PyPointCloudView createPointCloud( const std::vector<glm::vec3> & points,
                                                  const glm::vec3 &              color,
                                                  const float                    radius,
                                                  PyScene &                      scene );

        static PyPointCloudView createPointCloud( const std::vector<glm::vec3> & points,
                                                  const std::vector<float> &     radii,
                                                  const std::vector<glm::vec3> & colors,
                                                  PyScene &                      scene );

        static PyPointCloudView createPointCloud( const std::vector<glm::vec3> & points,
                                                  const float                    radius,
                                                  const glm::vec3 &              color,
                                                  PyScene &                      scene );

        static PyPointCloudView createPointCloud( const std::vector<glm::vec4> & points,
                                                  const std::vector<glm::vec4> & colors,
                                                  PyScene &                      scene );

        static PyPointCloudView createPointCloud( const std::vector<glm::vec4> & points,
                                                  const glm::vec4 &              color,
                                                  PyScene &                      scene );

        static PyPointCloudView createPointCloud( const std::vector<glm::vec4> & points,
                                                  const std::vector<glm::vec3> & colors,
                                                  PyScene &                      scene );

        static PyPointCloudView createPointCloud( const std::vector<glm::vec4> & points,
                                                  const glm::vec3 &              color,
                                                  PyScene &                      scene );
    };

    struct PyPointCloud
    {
        static constexpr auto in_place_delete = true;

        PyPointCloud()  = default;
        ~PyPointCloud() = default;

        PyPointCloud( const PyPointCloud & )             = delete;
        PyPointCloud & operator=( const PyPointCloud & ) = delete;

        PyPointCloud( PyPointCloud && other ) noexcept;
        PyPointCloud & operator=( PyPointCloud && other ) noexcept;

        std::vector<glm::vec4> points;
        std::vector<glm::vec4> pointsColors;

        bool needsUpdate = false;
        void update();

#if RVTX_GL
        gl::DebugPrimitivesHolder * holder;
#endif
    };
} // namespace rvtx

#endif