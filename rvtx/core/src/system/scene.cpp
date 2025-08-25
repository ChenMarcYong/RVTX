#include "rvtx/system/scene.hpp"

#include "rvtx/system/name.hpp"
#include "rvtx/system/visibility.hpp"

namespace rvtx
{
    entt::handle Scene::createEntity( std::string name, bool visible )
    {
        entt::entity entity = registry.create();
        registry.emplace<Name>( entity, std::move( name ) );
        registry.emplace<Visibility>( entity, visible );

        return { registry, entity };
    }

    std::vector<entt::handle> Scene::getEntitiesWithName( const std::string_view name )
    {
        std::vector<entt::handle> entities;
        const auto                view = registry.view<Name>();
        for ( const auto entity : view )
        {
            if ( name == view.get<Name>( entity ).name )
                entities.emplace_back( registry, entity );
        }

        return entities;
    }

    std::optional<entt::handle> Scene::getEntityWithName( const std::string_view name )
    {
        const auto view = registry.view<Name>();
        for ( const auto entity : view )
        {
            if ( name == view.get<Name>( entity ).name )
                return std::optional { entt::handle { registry, entity } };
        }

        return {};
    }
} // namespace rvtx
