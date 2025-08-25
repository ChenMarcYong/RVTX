#include "rvtx/optix/geometry/ses/ses_geometry.hpp"

namespace rvtx::optix
{
    inline Material & SesGeometry::getCustomMaterial() { return m_customMaterial; }
    inline void       SesGeometry::createCustomMaterial( const MaterialParameters & materialParameters )
    {
        m_customMaterial = materialParameters;
    }
    inline const Material & SesGeometry::getCustomMaterial() const { return m_customMaterial; }

    inline ColorMode   SesGeometry::getColorMode() const { return m_colorMode; }
    inline ColorMode & SesGeometry::getColorMode() { return m_colorMode; }
    inline void        SesGeometry::setColorMode( ColorMode colorMode ) { m_colorMode = colorMode; }
} // namespace rvtx::optix
