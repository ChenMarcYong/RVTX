#include "rvtx/gl/pass/post_process.hpp"

namespace rvtx::gl
{
    inline bool PostProcessPass::getEnableAO() const { return m_enableAO; }
    inline void PostProcessPass::setEnableAO( bool enable ) { m_enableAO = enable; }
    inline void PostProcessPass::toggleEnableAO() { m_enableAO = !m_enableAO; }

    inline GLuint PostProcessPass::getLinearizedDepthTexture() const { return m_linearizeDepth.getTexture(); }
    inline GLuint PostProcessPass::getSSAOTexture() const { return m_ssao.getTexture(); }
    inline GLuint PostProcessPass::getBlurTexture() const { return m_blur.getTexture(); }
    inline GLuint PostProcessPass::getShadingTexture() const { return m_shading.getTexture(); }
} // namespace rvtx::gl