#ifndef RVTX_GL_CORE_FWD_HPP
#define RVTX_GL_CORE_FWD_HPP

#include <cstdint>

// GL
using GLuint = uint32_t;
using GLint  = int32_t;
#define GL_INVALID_INDEX 0xFFFFFFFFu
#define GL_INVALID_VALUE 0x0501

// SDL2
using SDL_GLContext = void *;
struct SDL_Window;
using SDL_Event = union SDL_Event;

// VTX
namespace vtx
{
    // ui
    template<class Type>
    struct Span;

    template<class HandleType>
    struct HandleSpan;

    struct Aabb;

    struct Atom;
    struct Chain;
    struct Molecule;
} // namespace vtx

#endif // RVTX_GL_CORE_FWD_HPP
