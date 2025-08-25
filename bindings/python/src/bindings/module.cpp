#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings/defines.hpp"
#include "pyrvtx/py_scene.hpp"
#ifdef RVTX_GL
#include <rvtx/gl/window.hpp>

#include "pyrvtx/py_engine.hpp"
#endif

namespace rvtx
{
    RVTX_PY_DECLARE( Path );
    RVTX_PY_DECLARE( SplineType );
    RVTX_PY_DECLARE( Range );

    RVTX_PY_DECLARE_C( PyScene );

    RVTX_PY_DECLARE( AABB );
    RVTX_PY_DECLARE( Chain );
    RVTX_PY_DECLARE( Bond );
    RVTX_PY_DECLARE( Atom );
    RVTX_PY_DECLARE( Residue );

    RVTX_PY_DECLARE( PyCamera );
    RVTX_PY_DECLARE( PyGraphView );
    RVTX_PY_DECLARE( PyMolecule );
    RVTX_PY_DECLARE( PyMesh );
    RVTX_PY_DECLARE( PyPointCloud );

#ifdef RVTX_GL
    namespace gl
    {
        RVTX_PY_DECLARE( Window );
    }
    RVTX_PY_DECLARE( PyEngine );
#endif

    RVTX_PY_DECLARE( PathTimeInterpolator );
    RVTX_PY_DECLARE( PathKeyframeInterpolator );
    RVTX_PY_DECLARE( ProceduralMoleculeGenerator );
    RVTX_PY_DECLARE( Transform );

    NB_MODULE( rvtx, m )
    {
#ifdef NDEBUG // Remove leaked messages in 'Release' mode
        nb::set_leak_warnings(false);
#endif

        RVTX_PY_IMPORT( AABB );
        RVTX_PY_IMPORT( Transform );
        RVTX_PY_IMPORT( Range );

        nb::class_<PyScene> scene( m, "Scene" );
        RVTX_PY_MAIN_SCENE = PyScene();

        RVTX_PY_IMPORT( PyCamera );

        RVTX_PY_IMPORT( Atom );
        RVTX_PY_IMPORT( Chain );
        RVTX_PY_IMPORT( Residue );
        RVTX_PY_IMPORT( Bond );

        RVTX_PY_IMPORT( PyMolecule );
        RVTX_PY_IMPORT( PyMesh );
        RVTX_PY_IMPORT( PyPointCloud );

        RVTX_PY_IMPORT( PyGraphView );

        RVTX_PY_IMPORT( SplineType );

        RVTX_PY_IMPORT( Path );
        RVTX_PY_IMPORT( PathTimeInterpolator );
        RVTX_PY_IMPORT( PathKeyframeInterpolator );

        RVTX_PY_IMPORT( ProceduralMoleculeGenerator );

#ifdef RVTX_GL
        using namespace gl;
        RVTX_PY_IMPORT( Window );

        RVTX_PY_IMPORT( PyEngine );
#endif

        RVTX_PY_IMPORT_C(PyScene, scene);
    }
} // namespace rvtx