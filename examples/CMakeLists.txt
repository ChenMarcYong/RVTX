project(rVTX_Examples)

add_compile_options(-std=gnu++17 -D_GLIBCXX_USE_FLOAT128=0 -D__STDCPP_FLOAT128_T__=0)


define_cxx_flags(examples_compile_flags examples_compile_definition)
define_cuda_flags(examples_cuda_compile_flags)

add_dependency_folder(rVTX_Examples_Data "${CMAKE_CURRENT_SOURCE_DIR}/data" "${CMAKE_BINARY_DIR}/bin/data")

if (RVTX_INCLUDE_GL)
    #enable_language(CUDA)

    add_executable(        rVTX_GL_Example gl_example.cpp)
    target_link_libraries( rVTX_GL_Example PRIVATE rVTX::core rVTX::ui rVTX::gl)
    target_compile_options(rVTX_GL_Example PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${examples_cuda_compile_flags}>
    )
    target_compile_definitions(rVTX_GL_Example PRIVATE ${examples_compile_definition})
    target_compile_features(   rVTX_GL_Example PRIVATE cxx_std_20)

    add_dependencies(rVTX_GL_Example rVTX_Examples_Data)
endif()

if (RVTX_INCLUDE_GL AND RVTX_INCLUDE_OPTIX) # If RVTX_INCLUDE_OPTIX is 'ON', RVTX_INCLUDE_CUDA must be 'ON'
    enable_language(CUDA)

    add_executable(        rVTX_OptiX_Example optix_example.cpp)
    target_link_libraries( rVTX_OptiX_Example PRIVATE rVTX::core rVTX::ui rVTX::gl rVTX::raytracing rVTX::cuda)
    target_compile_options(rVTX_OptiX_Example PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${examples_compile_flags}>
            $<$<COMPILE_LANGUAGE:CUDA>:${examples_cuda_compile_flags}>
    )

    target_compile_definitions(rVTX_OptiX_Example PRIVATE ${examples_compile_definition})
    target_compile_features(   rVTX_OptiX_Example PRIVATE cxx_std_20)
    set_target_properties(     rVTX_OptiX_Example PROPERTIES
            CUDA_STANDARD 20
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    add_dependencies(rVTX_OptiX_Example rVTX_Examples_Data)
endif()


if (RVTX_INCLUDE_OPTIX) # If RVTX_INCLUDE_OPTIX is 'ON', RVTX_INCLUDE_CUDA must be 'ON'
    enable_language(CUDA)

    add_executable(        rVTX_OptiX_Headless_Example optix_headless_example.cpp)
    target_link_libraries( rVTX_OptiX_Headless_Example PRIVATE rVTX::core rVTX::gl rVTX::raytracing rVTX::cuda)
    target_compile_options(rVTX_OptiX_Headless_Example PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${examples_compile_flags}>
            $<$<COMPILE_LANGUAGE:CUDA>:${examples_cuda_compile_flags}>
    )

    target_compile_definitions(rVTX_OptiX_Headless_Example PRIVATE ${examples_compile_definition})
    target_compile_features(   rVTX_OptiX_Headless_Example PRIVATE cxx_std_20)
    set_target_properties(     rVTX_OptiX_Headless_Example PROPERTIES
            CUDA_STANDARD 20
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    add_dependencies(rVTX_OptiX_Headless_Example rVTX_Examples_Data)
endif()

