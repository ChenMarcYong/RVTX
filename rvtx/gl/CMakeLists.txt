define_cxx_flags(gl_compile_flags gl_compile_definition)
define_cuda_flags(gl_cuda_flags)

add_subdirectory(extern)

set(gl_includes
    include/rvtx/gl/core/fwd.hpp
    include/rvtx/gl/core/sampling.hpp

    include/rvtx/gl/system/debug_primitives.hpp

    include/rvtx/gl/geometry/ball_and_stick_geometry.hpp
    include/rvtx/gl/geometry/debug_primitives_geometry.hpp
    include/rvtx/gl/geometry/sphere_geometry.hpp
    include/rvtx/gl/geometry/sas_geometry.hpp
    include/rvtx/gl/geometry/sticks_geometry.hpp
    include/rvtx/gl/geometry/mesh_geometry.hpp
    include/rvtx/gl/geometry/handler.hpp

    include/rvtx/gl/pass/gbuffer.hpp
    include/rvtx/gl/pass/pass.hpp
    include/rvtx/gl/pass/post_process.hpp

    include/rvtx/gl/utils/buffer.hpp
    include/rvtx/gl/utils/buffer.inl
    include/rvtx/gl/utils/program.hpp
    include/rvtx/gl/utils/uniform.hpp
    include/rvtx/gl/utils/uniform.inl
    include/rvtx/gl/utils/snapshot.hpp

    include/rvtx/gl/renderer.hpp
    include/rvtx/gl/window.hpp
)
 
set(gl_sources
    src/gl/core/sampling.cpp

    src/gl/system/debug_primitives.cpp

    src/gl/pass/gbuffer.cpp
    src/gl/pass/pass.cpp
    src/gl/pass/post_process.cpp
    
    src/gl/geometry/ball_and_stick_geometry.cpp
    src/gl/geometry/debug_primitives_geometry.cpp
    src/gl/geometry/mesh_geometry.cpp
    src/gl/geometry/sphere_geometry.cpp
    src/gl/geometry/sas_geometry.cpp
    src/gl/geometry/sticks_geometry.cpp

    src/gl/utils/buffer.cpp
    src/gl/utils/program.cpp
    src/gl/utils/snapshot.cpp
    src/gl/utils/uniform.cpp

    src/gl/renderer.cpp
    src/gl/window.cpp
)

if (RVTX_INCLUDE_CUDA)
    set(gl_includes ${gl_includes}
            include/rvtx/gl/geometry/sesdf_geometry.hpp
        include/rvtx/gl/geometry/ssesdf_geometry.hpp
    )
    set(gl_sources ${gl_sources}
            src/gl/geometry/sesdf_geometry.cpp
        src/gl/geometry/ssesdf_geometry.cpp
    )
endif()

set(gl_extern_libraries ${gl_extern_libraries}
    rVTX::core
    rVTX::ui
)

if (RVTX_INCLUDE_CUDA)
    set(gl_extern_libraries ${gl_extern_libraries}
        rVTX::cuda
    )
endif()

add_library(			   rVTXgl STATIC ${gl_includes} ${gl_sources} ${gl_extern_sources})
target_include_directories(rVTXgl SYSTEM PUBLIC include/ ${gl_extern_includes})
target_include_directories(rVTXgl PUBLIC include/)
target_link_libraries(     rVTXgl PUBLIC ${gl_extern_libraries})
target_compile_features(   rVTXgl PRIVATE cxx_std_17)
target_compile_options(    rVTXgl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${gl_compile_flags}>)
target_compile_definitions(rVTXgl PRIVATE ${gl_compile_definition})

if (RVTX_INCLUDE_CUDA)
    set_target_properties(rVTXgl PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    define_cuda_flags(gl_cuda_flags)
    target_compile_options(rVTXgl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${gl_cuda_flags}> )

    get_compute_architecture(cuda_architecture)
    set_target_properties(rVTXgl PROPERTIES CUDA_ARCHITECTURES ${cuda_architecture})
endif()

add_dependency_folder(gl_shaders "${CMAKE_CURRENT_SOURCE_DIR}/shaders" "${CMAKE_BINARY_DIR}/bin/shaders")
add_dependencies(rVTXgl gl_shaders)

add_library(rVTX::gl ALIAS rVTXgl)