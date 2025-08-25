#ifndef RVTX_SYSTEM_SCENE_HPP
#define RVTX_SYSTEM_SCENE_HPP

#include <optional>
#include <string>

#include <entt/entity/handle.hpp>
#include <entt/entity/registry.hpp>

namespace rvtx
{
    struct Scene
    {
        entt::registry registry {};

        entt::handle                createEntity( std::string name = "", bool visible = true );
        std::vector<entt::handle>   getEntitiesWithName( const std::string_view name );
        std::optional<entt::handle> getEntityWithName( const std::string_view name );
    };
} // namespace rvtx

#endif // RVTX_SYSTEM_SCENE_HPP
