#include "rvtx/optix/environment.hpp"

namespace rvtx::optix
{
    inline Texture::View Environment::getEnvironmentView() const { return m_environment.getView(); }
    inline Texture::View Environment::getSamplingView() const { return m_sampling.getView(); }
} // namespace rvtx::optix