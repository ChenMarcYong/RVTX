#include <helper_math.h>

#include "rvtx/cuda/setup.cuh"
#include "rvtx/optix/texture.cuh"

// Strongly based on
// https://github.com/NVIDIA/cuda-samples/blob/3559ca4d088e12db33d6918621cab5c998ccecf1/Samples/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp#L101
namespace rvtx::optix
{
    static uint32_t getMipMapLevels( uint32_t width, uint32_t height )
    {
        std::size_t sz     = std::max( width, height );
        uint        levels = 0;
        while ( sz != 0 )
        {
            sz /= 2;
            levels++;
        }

        return levels;
    }

    __device__ __inline__ uchar4 to_uchar4( float4 vec )
    {
        return make_uchar4( (uint8_t)vec.x, (uint8_t)vec.y, (uint8_t)vec.z, (uint8_t)vec.w );
    }

    __global__ void d_mipmap( cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint imageW, uint imageH )
    {
        uint x = blockIdx.x * blockDim.x + threadIdx.x;
        uint y = blockIdx.y * blockDim.y + threadIdx.y;

        float px = 1.f / float( imageW );
        float py = 1.f / float( imageH );

        if ( ( x < imageW ) && ( y < imageH ) )
        {
            // take the average of 4 samples

            // we are using the normalized access to make sure non-power-of-two textures
            // behave well when downsized.
            float4 color = ( tex2D<float4>( mipInput, ( x + 0 ) * px, ( y + 0 ) * py ) )
                           + ( tex2D<float4>( mipInput, ( x + 1 ) * px, ( y + 0 ) * py ) )
                           + ( tex2D<float4>( mipInput, ( x + 1 ) * px, ( y + 1 ) * py ) )
                           + ( tex2D<float4>( mipInput, ( x + 0 ) * px, ( y + 1 ) * py ) );

            color /= 4.f;
            surf2Dwrite( color, mipOutput, x * sizeof( float4 ), y );
        }
    }

    void generateMipMaps( cudaMipmappedArray_t mipmapArray, cudaExtent size )
    {
        size_t width  = size.width;
        size_t height = size.height;
        uint   level  = 0;
        while ( width != 1 || height != 1 )
        {
            width /= 2;
            width = std::max( static_cast<std::size_t>( 1 ), width );
            height /= 2;
            height = std::max( static_cast<std::size_t>( 1 ), height );

            cudaArray_t levelFrom;
            cuda::cudaCheck( cudaGetMipmappedArrayLevel( &levelFrom, mipmapArray, level ) );
            cudaArray_t levelTo;
            cuda::cudaCheck( cudaGetMipmappedArrayLevel( &levelTo, mipmapArray, level + 1 ) );

            cudaExtent levelToSize = {};
            cuda::cudaCheck( cudaArrayGetInfo( nullptr, &levelToSize, nullptr, levelTo ) );
            assert( levelToSize.width == width );
            assert( levelToSize.height == height );
            assert( levelToSize.depth == 0 );

            // generate texture object for reading
            cudaTextureObject_t texInput;
            cudaResourceDesc    texRes = {};
            texRes.resType             = cudaResourceTypeArray;
            texRes.res.array.array     = levelFrom;

            cudaTextureDesc texDescr  = {};
            texDescr.normalizedCoords = 1;
            texDescr.filterMode       = cudaFilterModeLinear;

            texDescr.addressMode[ 0 ] = cudaAddressModeClamp;
            texDescr.addressMode[ 1 ] = cudaAddressModeClamp;
            texDescr.addressMode[ 2 ] = cudaAddressModeClamp;

            texDescr.readMode = cudaReadModeElementType;

            cuda::cudaCheck( cudaCreateTextureObject( &texInput, &texRes, &texDescr, NULL ) );

            // generate surface object for writing

            cudaSurfaceObject_t surfOutput;
            cudaResourceDesc    surfRes = {};
            surfRes.resType             = cudaResourceTypeArray;
            surfRes.res.array.array     = levelTo;

            cuda::cudaCheck( cudaCreateSurfaceObject( &surfOutput, &surfRes ) );

            // run mipmap kernel
            dim3 blockSize( 16, 16, 1 );
            dim3 gridSize(
                ( (uint)width + blockSize.x - 1 ) / blockSize.x, ( (uint)height + blockSize.y - 1 ) / blockSize.y, 1 );

            d_mipmap<<<gridSize, blockSize>>>( surfOutput, texInput, (uint)width, (uint)height );

            cuda::cudaCheck( cudaDeviceSynchronize() );
            cuda::cudaCheck( cudaGetLastError() );

            cuda::cudaCheck( cudaDestroySurfaceObject( surfOutput ) );
            cuda::cudaCheck( cudaDestroyTextureObject( texInput ) );

            level++;
        }
    }

    Texture Texture::From( uint32_t width, uint32_t height, ConstSpan<float> data )
    {
        Texture result {};
        result.m_width  = width;
        result.m_height = height;

        // how many mipmaps we need
        const uint32_t levels = getMipMapLevels( result.m_width, result.m_height );
        result.m_lodLevels      = std::max( 1.f, static_cast<float>( levels - 1 ) );

        // how many mipmaps we need
        cudaChannelFormatDesc desc   = cudaCreateChannelDesc<float4>();
        cudaExtent            extent = { width, height, 0 };
        cuda::cudaCheck( cudaMallocMipmappedArray( &result.m_mipmapArray, &desc, extent, levels ) );

        // upload level 0
        cudaArray_t level0;
        cuda::cudaCheck( cudaGetMipmappedArrayLevel( &level0, result.m_mipmapArray, 0 ) );

        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr            = make_cudaPitchedPtr( (void *)data.ptr, width * sizeof( float4 ), width, height );
        copyParams.dstArray          = level0;
        copyParams.extent            = extent;
        copyParams.extent.depth      = 1;
        copyParams.kind              = cudaMemcpyHostToDevice;
        cuda::cudaCheck( cudaMemcpy3D( &copyParams ) );

        // compute rest of mipmaps based on level 0
        generateMipMaps( result.m_mipmapArray, extent );

        // generate bindless texture object
        cudaResourceDesc resDescr  = {};
        resDescr.resType           = cudaResourceTypeMipmappedArray;
        resDescr.res.mipmap.mipmap = result.m_mipmapArray;

        cudaTextureDesc texDescr  = {};
        texDescr.normalizedCoords = 1;
        texDescr.filterMode       = cudaFilterModeLinear;
        texDescr.mipmapFilterMode = cudaFilterModeLinear;

        texDescr.addressMode[ 0 ] = cudaAddressModeClamp;
        texDescr.addressMode[ 1 ] = cudaAddressModeClamp;
        texDescr.addressMode[ 2 ] = cudaAddressModeClamp;

        texDescr.maxMipmapLevelClamp = float( levels - 1 );

        texDescr.readMode = cudaReadModeElementType;

        cuda::cudaCheck( cudaCreateTextureObject( &result.m_handle, &resDescr, &texDescr, nullptr ) );

        return result;
    }
} // namespace rvtx::optix