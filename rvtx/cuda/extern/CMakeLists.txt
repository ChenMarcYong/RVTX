add_subdirectory(cuda_helper)

# Prepare data for parent scope
set(cuda_extern_includes
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_helper
)

set(cuda_extern_libraries
    cuda_helper
)

set(cuda_extern_sources
)

if (RVTX_INCLUDE_GL)
    set(cuda_extern_libraries
        ${cuda_extern_libraries}
        gl3w
    )
endif()

set(cuda_extern_sources ${cuda_extern_sources} PARENT_SCOPE)
set(cuda_extern_includes ${cuda_extern_includes} PARENT_SCOPE)
set(cuda_extern_libraries ${cuda_extern_libraries} PARENT_SCOPE)
