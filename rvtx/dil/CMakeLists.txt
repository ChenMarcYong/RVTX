define_cxx_flags(dil_compile_flags dil_compile_definition)

set(dil_includes
    include/rvtx/dil/renderer.hpp
    include/rvtx/dil/utils/program.hpp
    include/rvtx/dil/utils/pipeline_manager.hpp
    include/rvtx/dil/utils/buffer.hpp

    include/rvtx/dil/geometry/handler.hpp
    include/rvtx/dil/geometry/sphere_handler.hpp
    include/rvtx/dil/geometry/sphere_holder.hpp
    include/rvtx/dil/geometry/ball_and_stick_handler.hpp
    include/rvtx/dil/geometry/ball_and_stick_holder.hpp


    include/rvtx/dil/pass/gbuffer.hpp
    include/rvtx/dil/pass/pass.hpp
    include/rvtx/dil/pass/post_process.hpp

    include/rvtx/dil/adapter/input_adapter.hpp
    include/rvtx/dil/adapter/diligent_input_adapter.hpp

    
)

set(dil_sources
    src/renderer.cpp
    src/utils/program.cpp

    src/geometry/sphere_handler.cpp
    src/geometry/ball_and_stick_handler.cpp
    src/geometry/ball_and_stick_holder.cpp
    src/geometry/sphere_holder.cpp
    

    src/utils/pipeline_manager.cpp
    src/utils/buffer.cpp

    src/pass/gbuffer.cpp
    src/pass/post_process.cpp
    src/pass/pass.cpp


    src/adapter/diligent_input_adapter.cpp
    

)

add_library(rVTXdil STATIC ${dil_includes} ${dil_sources})

target_include_directories(rVTXdil
    PUBLIC
        #${CMAKE_SOURCE_DIR}/rvtx/extern/diligentEngine/DiligentCore/install/include
        

        ${CMAKE_SOURCE_DIR}/rvtx/extern/diligentEngine/DiligentCore
        ${CMAKE_SOURCE_DIR}/rvtx/extern/diligentEngine/DiligentCore/Common/interface
        ${CMAKE_SOURCE_DIR}/rvtx/extern/diligentEngine/DiligentCore/Graphics/GraphicsEngine/interface
        ${CMAKE_SOURCE_DIR}/rvtx/extern/diligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface
        ${CMAKE_SOURCE_DIR}/rvtx/extern/diligentEngine/DiligentCore/Primitives/interface
        ${CMAKE_SOURCE_DIR}/rvtx/extern/diligentEngine/DiligentCore/Platforms/Basic/interface
        ${CMAKE_SOURCE_DIR}/rvtx/extern/diligentEngine/DiligentCore/Platforms/Win32/interface
        include/
)


set(dil_extern_libraries ${dil_extern_libraries}
        Diligent-SampleBase
        Diligent-BuildSettings
        Diligent-GraphicsEngine
        Diligent-GraphicsTools
        Diligent-Common
        Diligent-GraphicsAccessories
        rVTX::core
        rVTX::ui  
)

if (RVTX_INCLUDE_CUDA)
    set(dil_extern_libraries ${dil_extern_libraries}
        rVTX::cuda
    )
endif()


target_include_directories(rVTXdil PUBLIC include/)
target_link_libraries(     rVTXdil PUBLIC ${dil_extern_libraries})

target_compile_features(rVTXdil PRIVATE cxx_std_17)
target_compile_options(rVTXdil PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${dil_compile_flags}>)
target_compile_definitions(rVTXdil PRIVATE ${dil_compile_definition})

add_dependency_folder(
    hlsl_shaders
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders_hlsl"
    "${CMAKE_BINARY_DIR}/bin/shaders_hlsl"
)
add_dependencies(rVTXdil hlsl_shaders)

add_library(rVTX::dil ALIAS rVTXdil)
