#ifndef RVTX_SYSTEM_NAME_HPP
#define RVTX_SYSTEM_NAME_HPP

#include <string>

#include "entt/entt.hpp"

namespace rvtx
{
    struct Name
    {
        static constexpr bool in_place_delete = true;

        std::string name;
    };
} // namespace rvtx

#endif // RVTX_SYSTEM_NAME_HPP
