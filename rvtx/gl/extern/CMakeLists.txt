# Prepare data for parent scope
set(gl_extern_includes
    ${CMAKE_CURRENT_SOURCE_DIR}/stb
)

set(gl_extern_libraries
    SDL2::SDL2
    SDL2::SDL2main
    gl3w
    EnTT
)

set(gl_extern_sources
)

if (RVTX_INCLUDE_CUDA)
    set(gl_extern_libraries
        ${gl_extern_libraries}
        
        sesdf
    )
endif()

set(gl_extern_includes  ${gl_extern_includes}  PARENT_SCOPE)
set(gl_extern_sources   ${gl_extern_sources}   PARENT_SCOPE)
set(gl_extern_libraries ${gl_extern_libraries} PARENT_SCOPE)

