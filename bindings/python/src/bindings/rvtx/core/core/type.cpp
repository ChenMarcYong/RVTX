#include "rvtx/core/type.hpp"

#include "bindings/defines.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( Range )
    {
        nb::class_<Range>( m, "Range" )
            .def( nb::init<std::size_t, std::size_t>(), "start"_a = 0, "end"_a = 0 )
            .def_rw( "start", &Range::start )
            .def_rw( "end", &Range::end )
            .def_prop_ro( "size", &Range::size );
    }
} // namespace rvtx
