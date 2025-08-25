#ifndef RVTX_CUDA_BUFFER_CUH
#define RVTX_CUDA_BUFFER_CUH

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "rvtx/core/meta.hpp"
#include "rvtx/cuda/setup.cuh"

namespace rvtx::cuda
{
    class DeviceBuffer
    {
      public:
        template<class Type>
        static DeviceBuffer Typed( const std::size_t count, bool zeroInit = false )
        {
            return { count * sizeof( Type ), zeroInit };
        }

        DeviceBuffer() = default;
        DeviceBuffer( const std::size_t size, bool zeroInit = false );

        DeviceBuffer( const DeviceBuffer & )             = delete;
        DeviceBuffer & operator=( const DeviceBuffer & ) = delete;

        DeviceBuffer( DeviceBuffer && other ) noexcept;
        DeviceBuffer & operator=( DeviceBuffer && other ) noexcept;
        ~DeviceBuffer();

        void reset();

        inline uint8_t *       get();
        inline const uint8_t * get() const;

        template<class Type>
        Type * get( std::size_t offset = 0 );
        template<class Type>
        const Type * get( std::size_t offset = 0 ) const;

        operator bool() const;

        template<class Type>
        std::size_t size() const;
        std::size_t size() const;

        template<class Type>
        std::vector<Type> toHost();

      private:
        bool        m_initialized = false;
        std::size_t m_size        = 0;
        uint8_t *   m_ptr         = nullptr;
    };

} // namespace rvtx::cuda

#include "rvtx/cuda/buffer.inl"

#endif // RVTX_CUDA_BUFFER_CUH
