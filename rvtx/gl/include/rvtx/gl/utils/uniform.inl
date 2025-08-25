#include <cstring>
#include <type_traits>

#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "rvtx/gl/utils/uniform.hpp"

namespace rvtx::gl
{
    namespace detail
    {
        template<class Type>
        struct FalseType : public std::false_type
        {
        };
    } // namespace detail

    template<class Type>
    std::size_t UniformBuffer::computeNewOffset()
    {
        // std140 requires explicit alignment rules
        // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_uniform_buffer_object.txt
        std::size_t minAlignment = 0;

        constexpr bool IsScalar = std::is_arithmetic_v<std::remove_reference_t<Type>>;
        constexpr bool IsVec2   = std::is_same_v<glm::vec2, Type> || std::is_same_v<glm::ivec2, Type>;
        constexpr bool IsVec3   = std::is_same_v<glm::vec3, Type> || std::is_same_v<glm::ivec3, Type>;
        constexpr bool IsVec4   = std::is_same_v<glm::vec4, Type> || std::is_same_v<glm::ivec4, Type>;
        constexpr bool IsMat3   = std::is_same_v<glm::mat3, Type> || std::is_same_v<glm::dmat3, Type>;
        constexpr bool IsMat4   = std::is_same_v<glm::mat4, Type> || std::is_same_v<glm::dmat4, Type>;
        constexpr bool IsClass  = std::is_class_v<Type>;

        if constexpr ( IsScalar )
        {
            minAlignment = 4;
        }
        else if constexpr ( IsVec2 )
        {
            minAlignment = 2 * 4;
        }
        else if constexpr ( IsVec3 || IsVec4 || IsMat3 || IsMat4 || IsClass )
        {
            minAlignment = 4 * 4;
        }
        else
        {
            static_assert( detail::FalseType<Type>::value, "This type is not supported as uniform value." );
        }

        const std::size_t remaining = m_bufferUsedSize % minAlignment;
        return remaining == 0 ? remaining : minAlignment - remaining;
    }

    template<class Type>
    void UniformBuffer::addValue( const std::string & name, const Type & value )
    {
        if ( m_writingSave.find( name ) != m_writingSave.end() )
        {
            updateValue( name, value );
            return;
        }

        m_bufferUsedSize += computeNewOffset<Type>();

        constexpr std::size_t writeSize = sizeof( Type );
        if ( m_bufferCurrentMaxSize <= m_bufferUsedSize + writeSize )
        {
            m_bufferCurrentMaxSize <<= 1;
            m_uniformBuffer.resize( m_bufferCurrentMaxSize );
        }

        const auto data = m_uniformBuffer.scopedMap<uint8_t>( m_bufferUsedSize, writeSize, BufferAuthorization::Write );
        std::memcpy( data.get(), reinterpret_cast<const uint8_t *>( &value ), sizeof( Type ) );

        m_writingSave[ name ] = m_lastAdded = { m_bufferUsedSize, writeSize };
        m_bufferUsedSize += writeSize;
    }

    template<class Type>
    void UniformBuffer::updateValue( const std::string & name, const Type & value )
    {
        if ( m_writingSave.find( name ) == m_writingSave.end() )
        {
            addValue( name, value );
            return;
        }

        assert( sizeof( Type ) == m_writingSave[ name ].second
                && "Used Type size of Type  does not match with recorded Type size" );

        const auto data = m_uniformBuffer.scopedMap<uint8_t>(
            m_writingSave[ name ].first, m_writingSave[ name ].second, BufferAuthorization::Write );
        std::memcpy( data.get(), reinterpret_cast<const uint8_t *>( &value ), m_writingSave[ name ].second );
    }

} // namespace rvtx::gl
