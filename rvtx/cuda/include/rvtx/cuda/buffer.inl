#include "rvtx/cuda/buffer.cuh"
#include "rvtx/cuda/memory.cuh"

namespace rvtx::cuda
{
    inline uint8_t *       DeviceBuffer::get() { return m_ptr; }
    inline const uint8_t * DeviceBuffer::get() const { return m_ptr; }

    template<class Type>
    Type * DeviceBuffer::get( std::size_t offset )
    {
        return reinterpret_cast<Type *>( m_ptr + offset );
    }

    template<class Type>
    const Type * DeviceBuffer::get( std::size_t offset ) const
    {
        return reinterpret_cast<const Type *>( m_ptr + offset );
    }

    template<class Type>
    std::size_t DeviceBuffer::size() const
    {
        return m_size / sizeof( Type );
    }

    template<class Type>
    std::vector<Type> DeviceBuffer::toHost()
    {
        assert( m_size % sizeof( Type ) == 0 && "It seems that this type is not suitable." );

        const std::size_t hostBufferSize = m_size / sizeof( Type );
        std::vector<Type> buffer         = std::vector<Type>( hostBufferSize );

        mmemcpy<MemcpyType::DeviceToHost>( reinterpret_cast<uint8_t *>( buffer.data() ), m_ptr, m_size );
        return buffer;
    }
} // namespace rvtx::cuda
