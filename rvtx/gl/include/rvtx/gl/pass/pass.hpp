#ifndef RVTX_GL_PASS_PASS_HPP
#define RVTX_GL_PASS_PASS_HPP

#include <cstdint>

namespace rvtx::gl
{
    class Pass
    {
      public:
        Pass() = default;
        Pass( uint32_t width, uint32_t height );
        virtual ~Pass() = default;

        virtual void resize( uint32_t width, uint32_t height ) = 0;

      protected:
        uint32_t m_width;
        uint32_t m_height;
    };
} // namespace rvtx::gl

#endif // RVTX_GL_PASS_PASS_HPP
