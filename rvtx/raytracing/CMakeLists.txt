set(raytracing_libraries

)

if (RVTX_INCLUDE_OPTIX)
    define_cxx_flags(rt_optix_cxx_flags rt_optix_cxx_definition)
    define_cuda_flags(rt_optix_cuda_flags)

    add_subdirectory(extern)

    set(rt_optix_includes
            include/rvtx/optix/geometry/ball_and_stick/ball_and_stick_data.hpp
            include/rvtx/optix/geometry/ball_and_stick/ball_and_stick_geometry.cuh
            include/rvtx/optix/geometry/ball_and_stick/ball_and_stick_geometry.hpp
            include/rvtx/optix/geometry/ball_and_stick/ball_and_stick_handler.hpp
            include/rvtx/optix/geometry/ses/ses_data.hpp
            include/rvtx/optix/geometry/ses/ses_geometry.cuh
            include/rvtx/optix/geometry/ses/ses_geometry.hpp
            include/rvtx/optix/geometry/ses/ses_handler.hpp
            include/rvtx/optix/geometry/sphere/sphere_geometry.cuh
            include/rvtx/optix/geometry/sphere/sphere_geometry.hpp
            include/rvtx/optix/geometry/sphere/sphere_handler.hpp
            include/rvtx/optix/geometry/intersection.cuh
            include/rvtx/optix/geometry/multi_geometry_pipeline.hpp

            include/rvtx/optix/material/color.cuh
            include/rvtx/optix/material/lambertian.cuh
            include/rvtx/optix/material/material.cuh
            include/rvtx/optix/material/trowbridge_reitz_ggx.cuh

            include/rvtx/optix/context.cuh
            include/rvtx/optix/denoiser.hpp
            include/rvtx/optix/environment_sampler.cuh
            include/rvtx/optix/environment.hpp
            include/rvtx/optix/geometry.cuh
            include/rvtx/optix/pipeline.cuh
            include/rvtx/optix/program.cuh
            include/rvtx/optix/setup.cuh
            include/rvtx/optix/texture.cuh
    )

    set(rt_optix_sources
            src/optix/geometry/ball_and_stick/ball_and_stick_geometry.cpp
            src/optix/geometry/ball_and_stick/ball_and_stick_geometry.cu
            src/optix/geometry/ball_and_stick/ball_and_stick_handler.cpp
            src/optix/geometry/ses/ses_geometry.cpp
            src/optix/geometry/ses/ses_geometry.cu
            src/optix/geometry/ses/ses_handler.cpp
            src/optix/geometry/sphere/sphere_geometry.cpp
            src/optix/geometry/sphere/sphere_geometry.cu
            src/optix/geometry/sphere/sphere_handler.cpp
            src/optix/geometry/multi_geometry_pipeline.cu

            src/optix/context.cu
            src/optix/denoiser.cpp
            src/optix/environment_sampler.cu
            src/optix/environment.cpp
            src/optix/geometry.cu
            src/optix/pipeline.cu
            src/optix/program.cu
            src/optix/texture.cu
    )

    set(rt_optix_shaders
            shaders/rvtx/optix/path_tracer.cu

            shaders/rvtx/optix/geometry/ball_and_stick.cu

            shaders/rvtx/optix/geometry/ses/ses_circle.cu
            shaders/rvtx/optix/geometry/ses/ses_concave.cu
            shaders/rvtx/optix/geometry/ses/ses_convex.cu
            shaders/rvtx/optix/geometry/ses/ses_segment.cu

            shaders/rvtx/optix/geometry/sphere.cu
    )

    set(rt_optix_extern_libraries
            ${rt_optix_extern_libraries}
            rVTX::core
            rVTX::cuda
    )

    add_library(rVTX_RT_OptiX STATIC ${rt_optix_includes} ${rt_optix_sources} ${rt_optix_extern_sources})
    target_include_directories(rVTX_RT_OptiX SYSTEM PUBLIC ${rt_optix_extern_includes})
    target_include_directories(rVTX_RT_OptiX PUBLIC include/ shaders/)
    target_link_libraries(rVTX_RT_OptiX PUBLIC ${rt_optix_extern_libraries})
    target_compile_options(rVTX_RT_OptiX PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${rt_optix_cxx_flags}>
            $<$<COMPILE_LANGUAGE:CUDA>:${rt_optix_cuda_flags}>
    )

    target_compile_definitions(rVTX_RT_OptiX PRIVATE ${rt_optix_cxx_definition})
    target_compile_features(rVTX_RT_OptiX PRIVATE cxx_std_17)
    set_target_properties(rVTX_RT_OptiX PROPERTIES
            CUDA_STANDARD 17
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    get_compute_architecture(rt_optix_cuda_architecture)
    set_target_properties(rVTX_RT_OptiX PROPERTIES CUDA_ARCHITECTURES ${rt_optix_cuda_architecture})

    set(rt_optix_shaders_includes -Iinclude/ -Ishaders/)
    get_target_property(core_includes rVTX::core INCLUDE_DIRECTORIES)
    get_target_property(cuda_includes rVTX::cuda INCLUDE_DIRECTORIES)
    get_target_property(sesdf_includes sesdf INCLUDE_DIRECTORIES)
    foreach (includes IN LISTS rt_optix_extern_includes core_includes cuda_includes sesdf_includes)
        set(rt_optix_shaders_includes ${rt_optix_shaders_includes} -I${includes})
    endforeach ()

    compile_to_ptx(
            SOURCES ${rt_optix_shaders}
            TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ptx"
            GENERATED_FILES rt_optix_ptx_shaders
            NVCC_OPTIONS ${rt_optix_cuda_flags} ${rt_optix_shaders_includes} -std=c++17
    )

    add_custom_target(rt_optix_ptx_compilation DEPENDS ${rt_optix_ptx_shaders})
    add_dependencies(rVTX_RT_OptiX rt_optix_ptx_compilation)

    add_library(rVTX::raytracing::OptiX ALIAS rVTX_RT_OptiX)

    set(raytracing_libraries
            ${raytracing_libraries}

            rVTX_RT_OptiX
    )
endif (RVTX_INCLUDE_OPTIX)

# Do not create rVTX::raytracing if no API are selected
if (raytracing_libraries)
    add_library(rVTXraytracing INTERFACE)
    target_link_libraries(rVTXraytracing INTERFACE ${raytracing_libraries})
    add_library(rVTX::raytracing ALIAS rVTXraytracing)
endif (raytracing_libraries)