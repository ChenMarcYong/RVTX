add_subdirectory(nanobind)

# Prepare data for parent scope
set(bindings_python_extern_includes
)

set(bindings_python_extern_libraries
	rVTX::core
)

set(bindings_python_extern_sources
)


if (RVTX_INCLUDE_GL)
	set(bindings_python_extern_libraries ${bindings_python_extern_libraries}
		rVTX::gl
	)
endif()

if (RVTX_INCLUDE_CUDA)
	enable_language(CUDA)
	set(bindings_python_extern_libraries ${bindings_python_extern_libraries}
		rVTX::cuda
	)
endif()

if (RVTX_INCLUDE_OPTIX)
	set(bindings_python_extern_libraries ${bindings_python_extern_libraries}
		rVTX::cuda rVTX::raytracing::OptiX
	)
endif()

set(bindings_python_extern_sources ${bindings_python_extern_sources} PARENT_SCOPE)
set(bindings_python_extern_includes ${bindings_python_extern_includes} PARENT_SCOPE)
set(bindings_python_extern_libraries ${bindings_python_extern_libraries} PARENT_SCOPE)
