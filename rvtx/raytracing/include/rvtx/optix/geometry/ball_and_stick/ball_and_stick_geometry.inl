#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_geometry.hpp"

namespace rvtx::optix
{
    inline Material & BallAndStickGeometry::getCustomMaterial() { return m_customMaterial; }
    inline void       BallAndStickGeometry::createCustomMaterial( const MaterialParameters & materialParameters )
    {
        m_customMaterial = materialParameters;
    }
    inline const Material & BallAndStickGeometry::getCustomMaterial() const { return m_customMaterial; }

    inline ColorMode   BallAndStickGeometry::getColorMode() const { return m_colorMode; }
    inline ColorMode & BallAndStickGeometry::getColorMode() { return m_colorMode; }
    inline void        BallAndStickGeometry::setColorMode( ColorMode colorMode ) { m_colorMode = colorMode; }
} // namespace rvtx::optix
