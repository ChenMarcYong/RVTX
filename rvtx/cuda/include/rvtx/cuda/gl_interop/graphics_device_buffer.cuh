#ifndef RVTX_CUDA_BUFFER_CUH
#define RVTX_CUDA_BUFFER_CUH

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "rvtx/core/meta.hpp"
#include "rvtx/cuda/setup.cuh"

using GLuint = unsigned int;
#define GL_INVALID_VALUE 0x0501

namespace rvtx::cuda
{
    enum class AccessType : uint8_t
    {
        None      = 0,
        Read      = 1,
        Write     = 2,
        ReadWrite = 3
    };
    RVTX_DEFINE_ENUM_BITWISE_FUNCTION( AccessType )

    class GraphicsDeviceBuffer
    {
      public:
        template<class Type>
        static GraphicsDeviceBuffer Typed( std::size_t count,
                                           AccessType  access   = AccessType::ReadWrite,
                                           bool        zeroInit = false );

        GraphicsDeviceBuffer() = default;
        GraphicsDeviceBuffer( const std::size_t size,
                              AccessType        access   = AccessType::ReadWrite,
                              bool              zeroInit = false );

        GraphicsDeviceBuffer( GraphicsDeviceBuffer & other )                   = delete;
        GraphicsDeviceBuffer & operator=( const GraphicsDeviceBuffer & other ) = delete;

        GraphicsDeviceBuffer( GraphicsDeviceBuffer && other ) noexcept;
        GraphicsDeviceBuffer & operator=( GraphicsDeviceBuffer && other ) noexcept;

        virtual ~GraphicsDeviceBuffer();

        template<class Type>
        Type * map( std::size_t byteOffset = 0 );
        template<class Type>
        const Type * map( std::size_t byteOffset = 0 ) const;

        uint8_t * map( std::size_t byteOffset = 0 );
        uint8_t * map( std::size_t byteOffset = 0 ) const;

        void unmap() const;

        template<class Type>
        using ScopedMapping = std::unique_ptr<Type, std::function<void( Type * )>>;

        template<class Type>
        ScopedMapping<Type> scopedMap( std::size_t byteOffset = 0 );

        GLuint getId() const;

        std::size_t size() const;

        template<class Type>
        std::size_t size() const;

      private:
        std::size_t       m_size       = 0;
        mutable uint8_t * m_ptr        = nullptr;
        AccessType        m_accessType = AccessType::ReadWrite;

        // GL interoperability
        GLuint                         m_bufferId = GL_INVALID_VALUE;
        uint32_t                       m_type     = GL_INVALID_VALUE;
        mutable cudaGraphicsResource_t m_binding  = nullptr;
        cudaGraphicsRegisterFlags      m_registerFlags;
    };

} // namespace rvtx::cuda

#include "rvtx/cuda/gl_interop/graphics_device_buffer.inl"

#endif // RVTX_CUDA_BUFFER_CUH