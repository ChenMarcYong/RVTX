#include <cassert>

#include "rvtx/cuda/gl_interop/framebuffer.cuh"

#ifdef _WIN32
#include <Windows.h>
#endif // _WIN32

#include <GL/gl3w.h>
#include <cuda_gl_interop.h>

namespace rvtx::cuda
{
    DeviceFrameBuffer::DeviceFrameBuffer( uint32_t width, uint32_t height, uint8_t pixelSize, CUstream stream ) :
        m_stream( stream )
    {
        // Output dimensions must be at least 1 in both x and y to avoid an error with cudaMalloc.
        assert( width > 0 );
        assert( height > 0 );

        // Using GL Interop, expect that the active device is also the display device.
        int currentDevice;
        cudaCheck( cudaGetDevice( &currentDevice ) );

        int isDisplayDevice;
        cudaCheck( cudaDeviceGetAttribute( &isDisplayDevice, cudaDevAttrKernelExecTimeout, currentDevice ) );
        if ( !isDisplayDevice )
        {
            throw std::runtime_error(
                "GL interop is only available on display device, please use display device for optimal "
                "performance." );
        }

        resize( width, height, pixelSize );
    }

    DeviceFrameBuffer::~DeviceFrameBuffer()
    {
        try
        {
            if ( m_ptr )
                unmap();
            if ( m_bufferId != GL_INVALID_VALUE )
            {
                glBindBuffer( GL_ARRAY_BUFFER, 0 );
                glDeleteBuffers( 1, &m_bufferId );
            }
        }
        catch ( std::exception & e )
        {
            logger::error( "CUDAOutputBuffer destructor caught exception: {} ", e.what() );
        }
    }

    DeviceFrameBuffer::DeviceFrameBuffer( DeviceFrameBuffer && other )
    {
        std::swap( m_width, other.m_width );
        std::swap( m_height, other.m_height );
        std::swap( m_pixelSize, other.m_pixelSize );

        std::swap( m_binding, other.m_binding );
        std::swap( m_bufferId, other.m_bufferId );
        std::swap( m_ptr, other.m_ptr );
    }

    DeviceFrameBuffer & DeviceFrameBuffer::operator=( DeviceFrameBuffer && other )
    {
        std::swap( m_width, other.m_width );
        std::swap( m_height, other.m_height );
        std::swap( m_pixelSize, other.m_pixelSize );

        std::swap( m_binding, other.m_binding );
        std::swap( m_bufferId, other.m_bufferId );
        std::swap( m_ptr, other.m_ptr );

        return *this;
    }

    void DeviceFrameBuffer::resize( uint32_t width, uint32_t height, uint8_t pixelSize )
    {
        assert( width > 0 );
        assert( height > 0 );

        if ( m_width == width && m_height == height )
            return;

        if ( m_ptr )
            unmap();

        m_width     = width;
        m_height    = height;
        m_pixelSize = pixelSize;

        // GL buffer gets resized below
        glGenBuffers( 1, &m_bufferId );
        glBindBuffer( GL_PIXEL_PACK_BUFFER, m_bufferId );
        glBufferData(
            GL_PIXEL_PACK_BUFFER, static_cast<uint32_t>( m_pixelSize ) * m_width * m_height, nullptr, GL_STREAM_DRAW );
        glBindBuffer( GL_PIXEL_PACK_BUFFER, 0u );
        cudaGraphicsGLRegisterBuffer( &m_binding, m_bufferId, cudaGraphicsMapFlagsWriteDiscard );
    }

    uint8_t * DeviceFrameBuffer::map()
    {
        if ( !m_ptr )
        {
            std::size_t bufferSize = 0u;
            cudaCheck( cudaGraphicsMapResources( 1, &m_binding, m_stream ) );
            cudaCheck(
                cudaGraphicsResourceGetMappedPointer( reinterpret_cast<void **>( &m_ptr ), &bufferSize, m_binding ) );
        }

        return m_ptr;
    }

    const uint8_t * DeviceFrameBuffer::map() const
    {
        if ( !m_ptr )
        {
            std::size_t bufferSize = 0u;
            cudaCheck( cudaGraphicsMapResources( 1, &m_binding, m_stream ) );
            cudaCheck(
                cudaGraphicsResourceGetMappedPointer( reinterpret_cast<void **>( &m_ptr ), &bufferSize, m_binding ) );
        }

        return m_ptr;
    }

    void DeviceFrameBuffer::unmap()
    {
        m_ptr = nullptr;

        cudaCheck( cudaGraphicsUnmapResources( 1, &m_binding, m_stream ) );
    }
} // namespace rvtx::cuda
