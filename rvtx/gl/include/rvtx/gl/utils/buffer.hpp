#ifndef RVTX_GL_UTILS_BUFFER_HPP
#define RVTX_GL_UTILS_BUFFER_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "rvtx/core/meta.hpp"
#include "rvtx/core/type.hpp"
#include "rvtx/gl/core/fwd.hpp"

namespace rvtx::gl
{
    enum class BufferType : uint32_t
    {
        Uniform,
        SSBO,
        Array,
        ElementArray,
    };

    enum class BufferAuthorization : uint32_t
    {
        None       = 0,
        Dynamic    = 1,
        Read       = 2,
        Write      = 4,
        Persistent = 8,
    };

    RVTX_DEFINE_ENUM_BITWISE_FUNCTION( BufferAuthorization )

    class Buffer
    {
      public:
        template<class Type>
        static Buffer Typed( rvtx::ConstSpan<Type> data,
                             BufferAuthorization   authorization = BufferAuthorization::None );

        Buffer() = default;
        Buffer( rvtx::ConstSpan<uint8_t> data, BufferAuthorization authorization = BufferAuthorization::None );
        Buffer( std::size_t         allocationSize,
                BufferAuthorization authorization = BufferAuthorization::None,
                bool                zeroInit      = true );

        Buffer( const Buffer & )             = delete;
        Buffer & operator=( const Buffer & ) = delete;

        Buffer( Buffer && ) noexcept;
        Buffer & operator=( Buffer && ) noexcept;

        ~Buffer();

        GLuint                   getId() const;
        rvtx::HandleSpan<GLuint> view() const;
        void                     bind( uint32_t index, BufferType bufferType = BufferType::SSBO ) const;
        void bind( uint32_t index, uint32_t offset, uint32_t range, BufferType bufferType = BufferType::SSBO ) const;
        void bind( BufferType bufferType = BufferType::ElementArray ) const;
        void unbind( BufferType bufferType = BufferType::ElementArray ) const;

        template<class Type>
        Type *    map( std::size_t startingPoint, std::size_t mappingSize, BufferAuthorization mappingType );
        uint8_t * map( std::size_t startingPoint, std::size_t mappingSize, BufferAuthorization mappingType );
        void      unmap() const;

        template<class Type>
        using ScopedMapping = std::unique_ptr<Type, std::function<void( Type * )>>;

        template<class Type>
        ScopedMapping<Type> scopedMap( std::size_t         startingPoint,
                                       std::size_t         mappingSize,
                                       BufferAuthorization mappingType );

        void resize( std::size_t newSize, bool zeroInit = true );

      private:
        GLuint              m_id             = GL_INVALID_INDEX;
        std::size_t         m_size           = 0;
        BufferAuthorization m_authorizations = BufferAuthorization::None;
    };
} // namespace rvtx::gl

#include "rvtx/gl/utils/buffer.inl"

#endif // RVTX_GL_UTILS_BUFFER_HPP
