define_cxx_flags( cuda_compile_flags cuda_compile_definition)
define_cuda_flags(cuda_flags)

add_subdirectory(extern)

set(cuda_includes
    include/rvtx/cuda/benchmark.cuh
    include/rvtx/cuda/buffer.cuh
    include/rvtx/cuda/execution.cuh
    include/rvtx/cuda/math.cuh
    include/rvtx/cuda/memory.cuh
    include/rvtx/cuda/random.cuh
    include/rvtx/cuda/utility.cuh
    include/rvtx/cuda/setup.cuh
    include/rvtx/cuda/type.cuh
)

set(cuda_sources
    src/cuda/benchmark.cu
    src/cuda/buffer.cu
)

if (RVTX_INCLUDE_GL)
    set(cuda_includes
        ${cuda_includes}
        include/rvtx/cuda/gl_interop/framebuffer.cuh
        include/rvtx/cuda/gl_interop/graphics_device_buffer.cuh
    )

    set(cuda_sources
        ${cuda_sources}
        src/cuda/gl_interop/framebuffer.cu
        src/cuda/gl_interop/graphics_device_buffer.cu
    )
endif()

set(cuda_extern_libraries
    ${cuda_extern_libraries}
    rVTX::core
)

add_library(			   rVTXcuda STATIC ${cuda_includes} ${cuda_sources} ${cuda_extern_sources})
target_include_directories(rVTXcuda SYSTEM PUBLIC ${cuda_extern_includes})
target_include_directories(rVTXcuda PUBLIC include/)
target_link_libraries(     rVTXcuda PUBLIC ${cuda_extern_libraries})
target_compile_features(   rVTXcuda PRIVATE cxx_std_17)
target_compile_options(    rVTXcuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${cuda_compile_flags}>)
target_compile_definitions(rVTXcuda PRIVATE ${cuda_compile_definition})

set_target_properties(rVTXcuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
define_cuda_flags(cuda_flags)
target_compile_options(rVTXcuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}> )

get_compute_architecture(cuda_architecture)
set_target_properties(rVTXcuda PROPERTIES CUDA_ARCHITECTURES ${cuda_architecture})

add_library(rVTX::cuda ALIAS rVTXcuda)