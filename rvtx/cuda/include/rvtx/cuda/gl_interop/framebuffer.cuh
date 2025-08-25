#ifndef RVTX_CUDA_FRAMEBUFFER_CUH
#define RVTX_CUDA_FRAMEBUFFER_CUH

#include <cstdint>

#include "rvtx/cuda/buffer.cuh"
#include "rvtx/cuda/setup.cuh"

using GLuint = unsigned int;
#define GL_INVALID_VALUE 0x0501

namespace rvtx::cuda
{
    // Based on https://github.com/ingowald/optix7course and https://github.com/NVIDIA/OptiX_Apps
    class DeviceFrameBuffer
    {
      public:
        template<class Type>
        static DeviceFrameBuffer Typed( uint32_t width, uint32_t height, CUstream stream = 0 );

        DeviceFrameBuffer() = default;
        DeviceFrameBuffer( uint32_t width, uint32_t height, uint8_t pixelSize, CUstream stream = 0 );
        ~DeviceFrameBuffer();

        DeviceFrameBuffer( const DeviceFrameBuffer & other )             = delete;
        DeviceFrameBuffer & operator=( const DeviceFrameBuffer & other ) = delete;

        DeviceFrameBuffer( DeviceFrameBuffer && other );
        DeviceFrameBuffer & operator=( DeviceFrameBuffer && other );

        void resize( uint32_t width, uint32_t height, uint8_t pixelSize );

        template<class PixelType>
        void resize( uint32_t width, uint32_t height );

        uint8_t *       map();
        const uint8_t * map() const;
        template<class PixelType>
        PixelType * map();
        template<class PixelType>
        const PixelType * map() const;
        void              unmap();

        template<class Type>
        std::vector<Type> toHost();

        inline uint32_t getWidth() const;
        inline uint32_t getHeight() const;
        inline uint8_t  getPixelSize() const;
        inline uint32_t getSize() const;

        inline GLuint getId() const;

      private:
        CUstream m_stream;

        uint32_t m_width     = 0u;
        uint32_t m_height    = 0u;
        uint8_t  m_pixelSize = 0u;

        GLuint                         m_bufferId = GL_INVALID_VALUE;
        mutable cudaGraphicsResource * m_binding  = nullptr;
        mutable uint8_t *              m_ptr      = nullptr;
    };

} // namespace rvtx::cuda

#include "rvtx/cuda/gl_interop/framebuffer.inl"

#endif // RVTX_CUDA_FRAMEBUFFER_CUH
