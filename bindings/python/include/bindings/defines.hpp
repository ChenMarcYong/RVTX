#include "nanobind/nanobind.h"
#include "pyrvtx/py_glm.hpp"

namespace nb = nanobind;
using namespace nb::literals;

#define RVTX_PY_MAIN_SCENE m.attr( "main_scene" )

#define RVTX_PY_DECLARE( Name ) extern void python_export_##Name( nb::module_ & m )
#define RVTX_PY_EXPORT( Name ) void python_export_##Name( nb::module_ & m )
#define RVTX_PY_IMPORT( Name ) python_export_##Name( m )

#define RVTX_PY_DECLARE_C( Name ) extern void python_export_##Name( nb::class_<Name> & c )
#define RVTX_PY_EXPORT_C( Name ) void python_export_##Name( nb::class_<Name> & c )
#define RVTX_PY_IMPORT_C( Name, c ) python_export_##Name( c )