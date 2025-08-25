#ifndef RVTX_GL_UTILS_UNIFORM_HPP
#define RVTX_GL_UTILS_UNIFORM_HPP

#include <string>
#include <unordered_map>

#include "rvtx/gl/utils/buffer.hpp"

namespace rvtx::gl
{
    class Program;
    class UniformBuffer
    {
      public:
        UniformBuffer();

        // Offer to set binding point from UBO's name (see: glGetUniformBlockIndex)
        UniformBuffer( const Program & program, const std::string & bindingName, const uint32_t bindingPoint = 0 );

        UniformBuffer( const UniformBuffer & )             = delete;
        UniformBuffer & operator=( const UniformBuffer & ) = delete;

        UniformBuffer( UniformBuffer && ) noexcept             = default;
        UniformBuffer & operator=( UniformBuffer && ) noexcept = default;

        ~UniformBuffer() = default;

        void setBufferType( BufferType bindingType );
        void setBinding( uint32_t binding );

        template<class Type>
        void addValue( const std::string & name, const Type & value = Type {} );
        template<class Type>
        void updateValue( const std::string & name, const Type & value );

        void bind() const;

      private:
        template<class Type>
        std::size_t computeNewOffset();

        constexpr static std::size_t BufferInitializationSize = sizeof( float ) * 64;

        std::size_t m_bufferCurrentMaxSize;
        std::size_t m_bufferUsedSize = 0;

        Buffer     m_uniformBuffer;
        BufferType m_bindingType;

        uint32_t m_bindingPoint = 0;

        using AttributeInformation = std::pair<std::size_t /*offset*/, std::size_t /*padding*/>;
        AttributeInformation                                  m_lastAdded = { 0, 0 };
        std::unordered_map<std::string, AttributeInformation> m_writingSave;
    };

} // namespace rvtx::gl

#include "rvtx/gl/utils/uniform.inl"

#endif // RVTX_GL_UTILS_UNIFORM_HPP
