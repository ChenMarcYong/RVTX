#include "rvtx/cuda/gl_interop/graphics_device_buffer.cuh"

#ifdef _WIN32
#include <Windows.h>
#endif // _WIN32

#include <cassert>

#include <GL/gl3w.h>
#include <cuda_gl_interop.h>

namespace rvtx::cuda
{
    GraphicsDeviceBuffer::GraphicsDeviceBuffer( const std::size_t size, AccessType access, bool zeroInit ) :
        m_size( size ), m_registerFlags( cudaGraphicsRegisterFlagsNone )
    {
        assert( size > 0 && "Trying to allocate GraphicsDeviceBuffer of size 0" );
        glCreateBuffers( 1, &m_bufferId );

        GLsizeiptr accessMask = 0;
        if ( ( access & AccessType::Read ) != AccessType::None )
            accessMask |= GL_MAP_READ_BIT;
        if ( ( access & AccessType::Write ) != AccessType::None )
            accessMask |= GL_MAP_WRITE_BIT;

        glNamedBufferStorage( m_bufferId, size, nullptr, accessMask );
        cudaCheck( cudaGraphicsGLRegisterBuffer( &m_binding, m_bufferId, m_registerFlags ) );

        if ( zeroInit )
        {
            const auto data = scopedMap<uint8_t>();
            cudaCheck( cudaMemset( data.get(), 0, size ) );
        }
    }

    GraphicsDeviceBuffer::GraphicsDeviceBuffer( GraphicsDeviceBuffer && other ) noexcept :
        m_registerFlags( cudaGraphicsRegisterFlagsNone )
    {
        std::swap( m_size, other.m_size );
        std::swap( m_ptr, other.m_ptr );
        std::swap( m_accessType, other.m_accessType );
        std::swap( m_bufferId, other.m_bufferId );
        std::swap( m_type, other.m_type );
        std::swap( m_binding, other.m_binding );
        std::swap( m_registerFlags, other.m_registerFlags );
    }

    GraphicsDeviceBuffer & GraphicsDeviceBuffer::operator=( GraphicsDeviceBuffer && other ) noexcept
    {
        std::swap( m_size, other.m_size );
        std::swap( m_ptr, other.m_ptr );
        std::swap( m_accessType, other.m_accessType );
        std::swap( m_bufferId, other.m_bufferId );
        std::swap( m_type, other.m_type );
        std::swap( m_binding, other.m_binding );
        std::swap( m_registerFlags, other.m_registerFlags );

        return *this;
    }

    GraphicsDeviceBuffer::~GraphicsDeviceBuffer()
    {
        if ( !glIsBuffer( m_bufferId ) )
            return;

        unmap();
        cudaCheck( cudaGraphicsUnregisterResource( m_binding ) );
        glDeleteBuffers( 1, &m_bufferId );
    }

    uint8_t * GraphicsDeviceBuffer::map( std::size_t byteOffset )
    {
        if ( !m_ptr )
        {
            cudaCheck( cudaGraphicsMapResources( 1, &m_binding ) );
            cudaCheck(
                cudaGraphicsResourceGetMappedPointer( reinterpret_cast<void **>( &m_ptr ), nullptr, m_binding ) );
        }

        return m_ptr + byteOffset;
    }

    uint8_t * GraphicsDeviceBuffer::map( std::size_t byteOffset ) const
    {
        if ( !m_ptr )
        {
            cudaCheck( cudaGraphicsMapResources( 1, &m_binding ) );
            cudaCheck(
                cudaGraphicsResourceGetMappedPointer( reinterpret_cast<void **>( &m_ptr ), nullptr, m_binding ) );
        }

        return m_ptr + byteOffset;
    }

    void GraphicsDeviceBuffer::unmap() const
    {
        if ( !m_ptr )
            return;

        m_ptr = nullptr;
        cudaCheck( cudaGraphicsUnmapResources( 1, &m_binding ) );
    }
    GLuint GraphicsDeviceBuffer::getId() const { return m_bufferId; }

    std::size_t GraphicsDeviceBuffer::size() const { return m_size; }

} // namespace rvtx::cuda