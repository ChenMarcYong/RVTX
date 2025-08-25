add_subdirectory(extern)	


set(bindings_python_includes
	include/bindings/defines.hpp
	include/pyrvtx/py_camera.hpp
	include/pyrvtx/py_glm.hpp
	include/pyrvtx/py_molecule.hpp
	include/pyrvtx/py_mesh.hpp
	include/pyrvtx/py_scene.hpp
)
 
set(bindings_python_sources
	src/py_camera.cpp
	src/py_molecule.cpp
	src/py_mesh.cpp
	src/py_scene.cpp

	src/bindings/rvtx/core/core/path.cpp
	src/bindings/rvtx/core/core/type.cpp
	src/bindings/rvtx/core/core/spline.cpp
	src/bindings/rvtx/core/core/aabb.cpp

	src/bindings/pyrvtx/py_engine.cpp
	src/bindings/gl/window.cpp

	src/bindings/rvtx/core/molecule/molecule.cpp

	src/bindings/pyrvtx/py_camera.cpp
	src/bindings/pyrvtx/py_graph.cpp
	src/bindings/pyrvtx/py_molecule.cpp
	src/bindings/pyrvtx/py_mesh.cpp
	src/bindings/pyrvtx/py_point_cloud.cpp
	src/bindings/pyrvtx/py_scene.cpp

	src/bindings/rvtx/core/system/path_interpolator.cpp
	src/bindings/rvtx/core/system/procedural_molecule_generator.cpp
	src/bindings/rvtx/core/system/transform.cpp
)

if (RVTX_INCLUDE_GL)
	set(bindings_python_includes ${bindings_python_includes}
		include/pyrvtx/py_engine.hpp
		include/pyrvtx/py_point_cloud.hpp
		include/pyrvtx/py_graph.hpp
	)
	set(bindings_python_sources ${bindings_python_sources}
		src/py_engine.cpp
		src/py_point_cloud.cpp
		src/py_graph.cpp
	)
endif()

# Python bindings module
nanobind_add_module(rvtx src/bindings/module.cpp ${bindings_python_includes} ${bindings_python_sources} ${bindings_python_extern_includes} ${bindings_python_extern_sources})
target_link_libraries(rvtx PUBLIC ${bindings_python_extern_libraries} ${bindings_python_extern_sources})
target_include_directories(rvtx SYSTEM PUBLIC include/ ${bindings_python_extern_includes})
target_include_directories(rvtx PUBLIC include/ ${bindings_python_extern_includes})


set(python_library_includes
		include/pyrvtx/py_camera.hpp
		include/pyrvtx/py_molecule.hpp
		include/pyrvtx/py_scene.hpp
		include/pyrvtx/py_mesh.hpp
)

set(python_library_sources
		src/py_camera.cpp
		src/py_molecule.cpp
		src/py_scene.cpp
		src/py_mesh.cpp
)

if (RVTX_INCLUDE_GL)
	set(python_library_includes ${python_library_includes}
			include/pyrvtx/py_engine.hpp
			include/pyrvtx/py_point_cloud.hpp
			include/pyrvtx/py_graph.hpp
	)
	set(python_library_sources ${python_library_sources}
			src/py_engine.cpp
			src/py_point_cloud.cpp
			src/py_graph.cpp
	)
endif()

# Python full library
add_library(			   pyrVTX STATIC  ${bindings_python_includes} ${bindings_python_sources} ${bindings_python_extern_includes} ${bindings_python_extern_sources})
target_include_directories(pyrVTX SYSTEM PUBLIC include/ ${bindings_python_extern_includes} ${Python_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/extern/nanobind/include)
target_include_directories(pyrVTX PUBLIC include/ ${bindings_python_extern_includes})
target_link_libraries(     pyrVTX PUBLIC ${bindings_python_extern_libraries} ${bindings_python_extern_sources})
target_compile_features(   pyrVTX PRIVATE cxx_std_17)
target_compile_options(    pyrVTX PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${bindings_python_compile_flags}>)
target_compile_definitions(pyrVTX PRIVATE ${bindings_python_compile_definition})

# Python wrappers library
add_library(			   pyrVTXwrappers STATIC  ${python_library_includes} ${python_library_sources} ${bindings_python_extern_includes} ${bindings_python_extern_sources})
target_include_directories(pyrVTXwrappers SYSTEM PUBLIC include/ ${bindings_python_extern_includes})
target_include_directories(pyrVTXwrappers PUBLIC include/ ${bindings_python_extern_includes})
target_link_libraries(     pyrVTXwrappers PUBLIC ${bindings_python_extern_libraries} ${bindings_python_extern_sources})
target_compile_features(   pyrVTXwrappers PRIVATE cxx_std_17)
target_compile_options(    pyrVTXwrappers PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${bindings_python_compile_flags}>)
target_compile_definitions(pyrVTXwrappers PRIVATE ${bindings_python_compile_definition})

if (RVTX_INCLUDE_GL)
	target_compile_definitions(rvtx PUBLIC RVTX_GL)
	target_compile_definitions(pyrVTX PUBLIC RVTX_GL)
	target_compile_definitions(pyrVTXwrappers PUBLIC RVTX_GL)
endif()

if (RVTX_INCLUDE_CUDA)
	enable_language(CUDA)
	target_compile_definitions(rvtx   PUBLIC RVTX_CUDA)
	target_compile_definitions(pyrVTX PUBLIC RVTX_CUDA)
	target_compile_definitions(pyrVTXwrappers PUBLIC RVTX_CUDA)
endif()

if (RVTX_INCLUDE_OPTIX)
	target_compile_definitions(rvtx   PUBLIC RVTX_CUDA RVTX_OPTIX)
	target_compile_definitions(pyrVTX PUBLIC RVTX_CUDA RVTX_OPTIX)
	target_compile_definitions(pyrVTXwrappers PUBLIC RVTX_CUDA RVTX_OPTIX)
endif()

add_library(rVTX::bindings::python ALIAS pyrVTX)
add_library(rVTX::bindings::python::wrappers ALIAS pyrVTXwrappers)