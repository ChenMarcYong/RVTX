#ifndef RVTX_CORE_COLOR_HPP
#define RVTX_CORE_COLOR_HPP

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace rvtx
{
    struct Color
    {
        static constexpr glm::vec3 DarkRed { .84, 0, 0 };
        static constexpr glm::vec3 Red { 1, 0, 0 };
        static constexpr glm::vec3 DarkOrange { .7, .24, 0 };
        static constexpr glm::vec3 Orange { 1, .35, 0 };
        static constexpr glm::vec3 DarkYellow { .78, .58, 0 };
        static constexpr glm::vec3 Yellow { .97, .9, .04 };
        static constexpr glm::vec3 DarkGreen { 0, .23, 0 };
        static constexpr glm::vec3 Green { 0, .8, .05 };
        static constexpr glm::vec3 Cyan { 0, .96, 1 };
        static constexpr glm::vec3 Blue { 0, .19, 1 };
        static constexpr glm::vec3 DarkBlue { 0, .19, .47 };
        static constexpr glm::vec3 Plum { .29, 0, .25 };
        static constexpr glm::vec3 Purple { .59, 0, .52 };
        static constexpr glm::vec3 Pink { 1, .2, .65 };
        static constexpr glm::vec3 Brown { .55, .24, 0 };
        static constexpr glm::vec3 Black { 0, 0, 0 };
        static constexpr glm::vec3 LightGray { .76, .76, .76 };
        static constexpr glm::vec3 Gray { .5, .5, .5 };
        static constexpr glm::vec3 DarkGray { .32, .32, .32 };
        static constexpr glm::vec3 White { 1, 1, 1 };
        static constexpr glm::vec3 Beige { .988, 0.937, 0.729 };
    };

    struct Color4
    {
        static constexpr glm::vec4 DarkRed { .84, 0, 0, 1 };
        static constexpr glm::vec4 Red { 1, 0, 0, 1 };
        static constexpr glm::vec4 DarkOrange { .7, .24, 0, 1 };
        static constexpr glm::vec4 Orange { 1, .35, 0, 1 };
        static constexpr glm::vec4 DarkYellow { .78, .58, 0, 1 };
        static constexpr glm::vec4 Yellow { .97, .9, .04, 1 };
        static constexpr glm::vec4 DarkGreen { 0, .23, 0, 1 };
        static constexpr glm::vec4 Green { 0, .8, .05, 1 };
        static constexpr glm::vec4 Cyan { 0, .96, 1, 1 };
        static constexpr glm::vec4 Blue { 0, .19, 1, 1 };
        static constexpr glm::vec4 DarkBlue { 0, .19, .47, 1 };
        static constexpr glm::vec4 Plum { .29, 0, .25, 1 };
        static constexpr glm::vec4 Purple { .59, 0, .52, 1 };
        static constexpr glm::vec4 Pink { 1, .2, .65, 1 };
        static constexpr glm::vec4 Brown { .55, .24, 0, 1 };
        static constexpr glm::vec4 Black { 0, 0, 0, 1 };
        static constexpr glm::vec4 LightGray { .76, .76, .76, 1 };
        static constexpr glm::vec4 Gray { .5, .5, .5, 1 };
        static constexpr glm::vec4 DarkGray { .32, .32, .32, 1 };
        static constexpr glm::vec4 White { 1, 1, 1, 1 };
        static constexpr glm::vec4 Beige { .988, 0.937, 0.729, 1 };
    };

    constexpr inline float getLuminance( const glm::vec3 x );
} // namespace rvtx

#include "rvtx/core/color.inl"

#endif // RVTX_CORE_COLOR_HPP