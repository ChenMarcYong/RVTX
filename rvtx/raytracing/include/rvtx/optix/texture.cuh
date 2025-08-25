#ifndef RVTX_OPTIX_TEXTURE_CUH
#define RVTX_OPTIX_TEXTURE_CUH

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include "rvtx/core/type.hpp"

namespace rvtx::optix
{
    class Texture
    {
      public:
        static Texture From( uint32_t width, uint32_t height, ConstSpan<float> data );

        struct View
        {
            cudaTextureObject_t  handle;
            cudaMipmappedArray_t mipmapArray;

            uint32_t width;
            uint32_t height;
            float    lodLevels;

            template<class Type>
            __device__ Type get( float x, float y, float lod = 0.f ) const;
        };

        inline View getView() const;

        __host__ __device__ inline cudaTextureObject_t getHandle() const;
        __host__ __device__ inline float               getLodLevels() const;

      private:
        cudaTextureObject_t  m_handle;
        cudaMipmappedArray_t m_mipmapArray;

        uint32_t m_width;
        uint32_t m_height;
        float    m_lodLevels;
    };
} // namespace rvtx::optix

#include "rvtx/optix/texture.inl"

#endif // RVTX_OPTIX_TEXTURE_CUH