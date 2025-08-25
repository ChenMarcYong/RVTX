#include "rvtx/gl/utils/buffer.hpp"

#include <cassert>
#include <utility>

#include <GL/gl3w.h>

namespace rvtx::gl
{
    GLenum toGl( BufferType type )
    {
        switch ( type )
        {
        case BufferType::Uniform: return GL_UNIFORM_BUFFER;
        case BufferType::SSBO: return GL_SHADER_STORAGE_BUFFER;
        case BufferType::Array: return GL_ARRAY_BUFFER;
        case BufferType::ElementArray: return GL_ELEMENT_ARRAY_BUFFER;
        default:
        {
            assert( false && "Unkown buffer type" );
            return GL_SHADER_STORAGE_BUFFER;
        }
        }
    }

    GLbitfield toGl( BufferAuthorization authorization )
    {
        GLenum glAuthorization = GL_NONE;
        while ( authorization != BufferAuthorization::None )
        {
            GLenum currentFlag = GL_NONE;
            if ( ( authorization & BufferAuthorization::Dynamic ) != BufferAuthorization::None )
            {
                currentFlag   = GL_DYNAMIC_STORAGE_BIT;
                authorization = authorization & ~BufferAuthorization::Dynamic;
            }
            else if ( ( authorization & BufferAuthorization::Read ) != BufferAuthorization::None )
            {
                currentFlag   = GL_MAP_READ_BIT;
                authorization = authorization & ~BufferAuthorization::Read;
            }
            else if ( ( authorization & BufferAuthorization::Write ) != BufferAuthorization::None )
            {
                currentFlag   = GL_MAP_WRITE_BIT;
                authorization = authorization & ~BufferAuthorization::Write;
            }
            else if ( ( authorization & BufferAuthorization::Persistent ) != BufferAuthorization::None )
            {
                currentFlag   = GL_MAP_PERSISTENT_BIT;
                authorization = authorization & ~BufferAuthorization::Persistent;
            }

            glAuthorization |= currentFlag;
        }

        return glAuthorization;
    }

    Buffer::Buffer( rvtx::ConstSpan<uint8_t> data, BufferAuthorization authorization ) :
        m_size( data.size ), m_authorizations( authorization )
    {
        glCreateBuffers( 1, &m_id );
        glNamedBufferStorage( m_id, data.size, data.ptr, toGl( authorization ) );
    }

    Buffer::Buffer( std::size_t allocationSize, BufferAuthorization authorization, bool zeroInit ) :
        m_size( allocationSize ), m_authorizations( authorization )
    {
        glCreateBuffers( 1, &m_id );
        if ( zeroInit )
        {
            const auto zeroData = std::vector<uint8_t>( m_size, 0u );
            glNamedBufferStorage( m_id, zeroData.size(), zeroData.data(), toGl( authorization ) );
        }
        else
        {
            glNamedBufferStorage( m_id, allocationSize, nullptr, toGl( authorization ) );
        }
    }

    Buffer::Buffer( Buffer && other ) noexcept
    {
        std::swap( m_id, other.m_id );
        std::swap( m_size, other.m_size );
        std::swap( m_authorizations, other.m_authorizations );
    }

    Buffer & Buffer::operator=( Buffer && other ) noexcept
    {
        std::swap( m_id, other.m_id );
        std::swap( m_size, other.m_size );
        std::swap( m_authorizations, other.m_authorizations );

        return *this;
    }

    Buffer::~Buffer()
    {
        if ( glIsBuffer( m_id ) )
            glDeleteBuffers( 1, &m_id );
    }

    GLuint Buffer::getId() const { return m_id; }

    rvtx::HandleSpan<GLuint> Buffer::view() const { return { m_id, 0, m_size }; }

    void Buffer::bind( uint32_t index, const BufferType bufferType ) const
    {
        glBindBufferBase( toGl( bufferType ), index, m_id );
    }

    void Buffer::bind( uint32_t index, uint32_t offset, uint32_t range, BufferType bufferType ) const
    {
        glBindBufferRange( toGl( bufferType ), index, m_id, offset, range );
    }

    void Buffer::bind( BufferType bufferType ) const { glBindBuffer( toGl( bufferType ), m_id ); }

    void Buffer::unbind( BufferType bufferType ) const { glBindBuffer( toGl( bufferType ), 0 ); }

    void Buffer::resize( const std::size_t newSize, const bool zeroInit )
    {
        GLuint newHandle;
        glCreateBuffers( 1, &newHandle );

        if ( zeroInit )
        {
            const auto zeroData = std::vector<uint8_t>( newSize, 0u );
            glNamedBufferStorage( newHandle, zeroData.size(), zeroData.data(), toGl( m_authorizations ) );
        }
        else
        {
            glNamedBufferStorage( newHandle, newSize, nullptr, toGl( m_authorizations ) );
        }

        glCopyNamedBufferSubData( m_id, newHandle, 0, 0, m_size );
        m_size = newSize;

        glDeleteBuffers( 1, &m_id );
        m_id = newHandle;
    }

    uint8_t * Buffer::map( std::size_t startingPoint, std::size_t mappingSize, BufferAuthorization mappingType )
    {
        assert( startingPoint <= m_size && "This mapping try to access out of ot the bound of the buffer." );
        assert( mappingSize <= m_size && "This mapping try to access out of ot the bound of the buffer." );

        return static_cast<uint8_t *>( glMapNamedBufferRange( m_id, startingPoint, mappingSize, toGl( mappingType ) ) );
    }
    void Buffer::unmap() const { glUnmapNamedBuffer( m_id ); }

} // namespace rvtx::gl
