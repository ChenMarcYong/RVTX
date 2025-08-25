#include "rvtx/optix/geometry/sphere/sphere_geometry.hpp"

namespace rvtx::optix
{
    inline Material & SphereGeometry::getCustomMaterial() { return m_customMaterial; }

    inline void SphereGeometry::createCustomMaterial( const MaterialParameters & materialParameters )
    {
        m_customMaterial = materialParameters;
    }

    inline const Material & SphereGeometry::getCustomMaterial() const { return m_customMaterial; }

    inline ColorMode   SphereGeometry::getColorMode() const { return m_colorMode; }
    inline ColorMode & SphereGeometry::getColorMode() { return m_colorMode; }
    inline void        SphereGeometry::setColorMode( ColorMode colorMode ) { m_colorMode = colorMode; }
} // namespace rvtx::optix
