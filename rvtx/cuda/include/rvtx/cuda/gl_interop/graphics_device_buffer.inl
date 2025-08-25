#include "rvtx/cuda/gl_interop/graphics_device_buffer.cuh"

namespace rvtx::cuda
{
    template<class Type>
    GraphicsDeviceBuffer GraphicsDeviceBuffer::Typed( const std::size_t count, AccessType access, bool zeroInit )
    {
        return GraphicsDeviceBuffer( count * sizeof( Type ), access, zeroInit );
    }

    template<class Type>
    Type * GraphicsDeviceBuffer::map( std::size_t byteOffset )
    {
        return reinterpret_cast<Type *>( map( byteOffset ) );
    }

    template<class Type>
    const Type * GraphicsDeviceBuffer::map( std::size_t byteOffset ) const
    {
        return reinterpret_cast<const Type *>( map( byteOffset ) );
    }

    template<class Type>
    GraphicsDeviceBuffer::ScopedMapping<Type> GraphicsDeviceBuffer::scopedMap( std::size_t byteOffset )
    {
        Type * ptr = map<Type>( byteOffset );
        return ScopedMapping<Type>( ptr, [ this ]( Type * ) { unmap(); } );
    }

    template<class Type>
    std::size_t GraphicsDeviceBuffer::size() const
    {
        return m_size / sizeof( Type );
    }
} // namespace rvtx::cuda