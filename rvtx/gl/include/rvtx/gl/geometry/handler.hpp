#ifndef RVTX_GL_GEOMETRY_HANDLER_HPP
#define RVTX_GL_GEOMETRY_HANDLER_HPP

#include <memory>
#include <type_traits>
#include <vector>

namespace rvtx
{
    class Scene;
    struct Camera;
} // namespace rvtx

namespace rvtx::gl
{
    // Called during GBuffer filling
    // Attachment template:
    //  // 3 16 bits for position.
    //  // 3 16 bits for normal.
    //  // 1 32 bits for padding.
    //  layout( location = 0 ) out uvec4 outViewPositionNormal;
    //  // 3 32 bits for color.
    //  // 1 32 bits for specular.
    //  layout( location = 1 ) out vec4 outColor;
    struct GeometryHandler
    {
        virtual void render( const Camera &, const Scene & ) {}

        virtual ~GeometryHandler() = default;
    };

    struct GeometryForwarder : public GeometryHandler
    {
        std::vector<std::unique_ptr<GeometryHandler>> handlers;

        template<class DerivedHandler, class... Args>
        DerivedHandler & add( Args &&... args )
        {
            static_assert( std::is_base_of<GeometryHandler, DerivedHandler>::value,
                           "DerivedHandler must be based on GeometryHandler." );

            auto             derivedHandler = std::make_unique<DerivedHandler>( std::forward<Args>( args )... );
            DerivedHandler * ptr            = derivedHandler.get();
            handlers.emplace_back( std::move( derivedHandler ) );
            return *ptr;
        }

        void clear() { handlers.clear(); }

        void render( const Camera & camera, const Scene & scene ) override
        {
            for ( auto & handler : handlers )
                handler->render( camera, scene );
        }
    };
} // namespace rvtx::gl

#endif // RVTX_GL_GEOMETRY_HANDLER_HPP
