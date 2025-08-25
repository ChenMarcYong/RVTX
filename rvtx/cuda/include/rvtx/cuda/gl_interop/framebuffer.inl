#include "rvtx/cuda/gl_interop/framebuffer.cuh"

namespace rvtx::cuda
{
    template<class Type>
    DeviceFrameBuffer DeviceFrameBuffer::Typed( uint32_t width, uint32_t height, CUstream stream )
    {
        return { width, height, sizeof( Type ), stream };
    }

    template<class PixelType>
    PixelType * DeviceFrameBuffer::map()
    {
        return reinterpret_cast<PixelType *>( map() );
    }

    template<class PixelType>
    const PixelType * DeviceFrameBuffer::map() const
    {
        return reinterpret_cast<const PixelType *>( map() );
    }

    template<class PixelType>
    void DeviceFrameBuffer::resize( uint32_t width, uint32_t height )
    {
        resize( width, height, sizeof( PixelType ) );
    }

    template<typename PixelType>
    std::vector<PixelType> DeviceFrameBuffer::toHost()
    {
        assert( sizeof( PixelType ) == m_pixelSize );

        std::vector<PixelType> pixels {};
        pixels.resize( m_width * m_height );

        cudaCheck(
            cudaMemcpy( pixels.data(), map(), m_width * m_height * sizeof( PixelType ), cudaMemcpyDeviceToHost ) );
        unmap();

        return pixels;
    }

    inline uint32_t DeviceFrameBuffer::getWidth() const { return m_width; }
    inline uint32_t DeviceFrameBuffer::getHeight() const { return m_height; }
    inline uint8_t  DeviceFrameBuffer::getPixelSize() const { return m_pixelSize; }
    inline uint32_t DeviceFrameBuffer::getSize() const { return m_width * m_height * m_pixelSize; }
    inline GLuint   DeviceFrameBuffer::getId() const { return m_bufferId; }
} // namespace rvtx::cuda
