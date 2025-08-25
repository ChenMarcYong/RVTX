#ifndef RVTX_UX_INPUT_HPP
#define RVTX_UX_INPUT_HPP

#include <optional>
#include <set>

#include <glm/vec2.hpp>

#include "rvtx/core/filesystem.hpp"

namespace rvtx
{
    enum class KeyAction
    {
        // Can be hold
        Pressed,

        // Only a single time
        Down,
        Up
    };

    // see: https://github.com/LuaDist/sdl/blob/master/include/SDL_scancode.h
    enum class Key
    {
        Unknown,

        A,
        B,
        C,
        D,
        E,
        F,
        G,
        H,
        I,
        J,
        K,
        L,
        M,
        N,
        O,
        P,
        Q,
        R,
        S,
        T,
        U,
        V,
        W,
        X,
        Y,
        Z,

        Return,
        Escape,
        BackSpace,
        Tab,
        Space,

        F1,
        F2,
        F3,
        F4,
        F5,
        F6,
        F7,
        F8,
        F9,
        F10,
        F11,
        F12,

        Right,
        Left,
        Down,
        Up,

        LCtrl,
        LShift,
        LAlt, /**< alt, option */
        LGui, /**< windows, command (apple), meta */
        RCtrl,
        RShift,
        RAlt, /**< alt gr, option */
        RGui, /**< windows, command (apple), meta */
    };

    struct Input
    {
        glm::uvec2 windowSize;
        bool       windowResized = false;

        float deltaTime;

        glm::ivec2 mousePosition;
        glm::ivec2 deltaMousePosition;

        int32_t deltaMouseWheel;

        bool doubleLeftClick = false;

        bool mouseLeftClicked = false;
        bool mouseLeftPressed = false;

        bool mouseRightClicked = false;
        bool mouseRightPressed = false;

        bool mouseMiddleClicked = false;
        bool mouseMiddlePressed = false;

        std::optional<std::filesystem::path> droppedFile;

        std::set<Key> keysPressed;
        std::set<Key> keysDown;
        std::set<Key> keysUp;

        bool isKeyPressed( const Key key ) const;
        bool isKeyDown( const Key key ) const;
        bool isKeyUp( const Key key ) const;

        bool isKeyActivated( const Key key, const KeyAction action ) const;

        void reset();
    };
} // namespace rvtx

#endif // RVTX_UX_INPUT_HPP
