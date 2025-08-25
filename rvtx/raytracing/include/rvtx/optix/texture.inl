#include "rvtx/optix/texture.cuh"

namespace rvtx::optix
{
    inline Texture::View Texture::getView() const
    {
        return { m_handle, m_mipmapArray, m_width, m_height, m_lodLevels };
    }

    template<class Type>
    __device__ Type Texture::View::get( float u, float v, float lod ) const
    {
#ifdef __CUDACC__
        return tex2DLod<Type>( handle, u, v, lod );
#else
        return Type {};
#endif // __CUDACC__
    }

    __host__ __device__ inline cudaTextureObject_t Texture::getHandle() const { return m_handle; }
    __host__ __device__ inline float               Texture::getLodLevels() const { return m_lodLevels; }
} // namespace rvtx::optix