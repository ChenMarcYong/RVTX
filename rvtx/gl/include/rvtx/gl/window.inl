#include "rvtx/gl/window.hpp"

namespace rvtx::gl
{
    inline SDL_Window *  Window::getHandle() { return m_window; }
    inline SDL_GLContext Window::getContext() { return m_glContext; }
    inline uint32_t      Window::getWidth() const { return m_width; }
    inline uint32_t      Window::getHeight() const { return m_height; }
    inline const Input & Window::getInput() const { return m_input; }
} // namespace rvtx::gl
