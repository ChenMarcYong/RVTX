#include "rvtx/core/spline.hpp"

#include "bindings/defines.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( SplineType )
    {
        nb::enum_<SplineType>( m, "SplineType" )
            .value( "CatmullRom", SplineType::CatmullRom )
            .value( "Linear", SplineType::Linear )
            .export_values();
    }
} // namespace rvtx
