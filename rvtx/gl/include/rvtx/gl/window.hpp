#ifndef RVTX_GL_WINDOW_HPP
#define RVTX_GL_WINDOW_HPP

#include <cstddef>
#include <string>

#include "rvtx/ux/input.hpp"

// GL
using GLuint = uint32_t;
using GLint  = int32_t;
#define GL_INVALID_INDEX 0xFFFFFFFFu
#define GL_INVALID_VALUE 0x0501

// SDL2
using SDL_GLContext = void *;
struct SDL_Window;
using SDL_Event = union SDL_Event;

namespace rvtx::gl
{
    class Window
    {
      public:
        Window( std::string title, std::size_t width = 1280, std::size_t height = 720, bool shown = true );

        Window( const Window & )             = delete;
        Window & operator=( const Window & ) = delete;

        Window( Window && other ) noexcept;
        Window & operator=( Window && other ) noexcept;

        ~Window();

        bool update();

        void resize( std::size_t width, std::size_t height );

        inline SDL_Window *  getHandle();
        inline SDL_GLContext getContext();
        inline uint32_t      getWidth() const;
        inline uint32_t      getHeight() const;
        inline const Input & getInput() const;

      private:
        std::string m_title;
        uint32_t    m_width;
        uint32_t    m_height;

        SDL_Window *  m_window    = nullptr;
        SDL_GLContext m_glContext = nullptr;

        Input    m_input {};
        uint64_t m_lastTimeStep = 0;
        bool     m_isVisible    = true;
    };
} // namespace rvtx::gl

#include "rvtx/gl/window.inl"

#endif // RVTX_UI_WINDOW_HPP
