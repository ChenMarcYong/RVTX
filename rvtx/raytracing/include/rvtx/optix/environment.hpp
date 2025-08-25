#ifndef RVTX_OPTIX_ENVIRONMENT_HPP
#define RVTX_OPTIX_ENVIRONMENT_HPP

#include "rvtx/core/filesystem.hpp"
#include "rvtx/optix/texture.cuh"

namespace rvtx::optix
{
    class Environment
    {
      public:
        Environment( std::filesystem::path path, float weight = 1.f );

        inline Texture::View getEnvironmentView() const;
        inline Texture::View getSamplingView() const;

      private:
        Texture m_environment;
        Texture m_sampling;
    };
} // namespace rvtx::optix

#include "rvtx/optix/environment.inl"

#endif // RVTX_OPTIX_ENVIRONMENT_HPP