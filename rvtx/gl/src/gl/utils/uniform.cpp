#include "rvtx/gl/utils/uniform.hpp"

#include <cassert>

#include <GL/gl3w.h>

#include "rvtx/gl/utils/program.hpp"

namespace rvtx::gl
{
    UniformBuffer::UniformBuffer() :
        m_bufferCurrentMaxSize( BufferInitializationSize ),
        m_uniformBuffer( m_bufferCurrentMaxSize, BufferAuthorization::Write ), m_bindingType( BufferType::Uniform )
    {
    }

    UniformBuffer::UniformBuffer( const Program &     program,
                                  const std::string & bindingName,
                                  const uint32_t      bindingPoint ) :
        m_bufferCurrentMaxSize( BufferInitializationSize ),
        m_uniformBuffer( m_bufferCurrentMaxSize, BufferAuthorization::Write ), m_bindingType( BufferType::Uniform )
    {
        const uint32_t index = glGetUniformBlockIndex( program.getId(), bindingName.c_str() );
        glUniformBlockBinding( program.getId(), index, static_cast<GLuint>( bindingPoint ) );
    }

    void UniformBuffer::setBufferType( BufferType bindingType ) { m_bindingType = bindingType; }
    void UniformBuffer::setBinding( uint32_t binding ) { m_bindingPoint = binding; }
    void UniformBuffer::bind() const { m_uniformBuffer.bind( m_bindingPoint, m_bindingType ); }
} // namespace rvtx::gl
