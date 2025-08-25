define_cxx_flags(core_compile_flags core_compile_definition)

add_subdirectory(extern)

set(core_includes
    include/rvtx/core/aabb.hpp
    include/rvtx/core/benchmark.hpp
    include/rvtx/core/color.hpp
    include/rvtx/core/filesystem.hpp
    include/rvtx/core/image.hpp
    include/rvtx/core/logger.hpp
    include/rvtx/core/math.hpp
    include/rvtx/core/meta.hpp
    include/rvtx/core/path.hpp
    include/rvtx/core/random.hpp
    include/rvtx/core/spline.hpp
    include/rvtx/core/spline_helpers.hpp
    include/rvtx/core/time.hpp
    include/rvtx/core/type.hpp

    include/rvtx/mesh/loader.hpp
    include/rvtx/mesh/mesh.hpp

    include/rvtx/molecule/color.hpp
    include/rvtx/molecule/loader.hpp
    include/rvtx/molecule/molecule.hpp

    include/rvtx/system/acceleration_structure.hpp
    include/rvtx/system/path_interpolator.hpp
    include/rvtx/system/camera.hpp
    include/rvtx/system/material_parameters.hpp
    include/rvtx/system/molecule_ids.hpp
    include/rvtx/system/scene.hpp
    include/rvtx/system/scene_descriptor.hpp
    include/rvtx/system/serialization.hpp
    include/rvtx/system/transform.hpp
    include/rvtx/system/procedural_molecule_generator.hpp
    include/rvtx/system/visibility.hpp
    
    include/rvtx/ux/input.hpp
    include/rvtx/ux/camera_controller.hpp
    include/rvtx/ux/controller.hpp
)

set(core_sources
    src/core/benchmark.cpp
    src/core/filesystem.cpp
    src/core/image.cpp
    src/core/logger.cpp
    src/core/math.cpp
    src/core/time.cpp

    src/mesh/loader.cpp
    src/mesh/mesh.cpp

    src/molecule/color.cpp
    src/molecule/loader.cpp
    src/molecule/molecule.cpp

    src/system/acceleration_structure.cpp
    src/system/camera.cpp
    src/system/scene.cpp
    src/system/scene_descriptor.cpp
    src/system/molecule_ids.cpp
    src/system/transform.cpp
    src/system/procedural_molecule_generator.cpp

    src/ux/camera_controller.cpp
    src/ux/input.cpp
)

add_library(			   rVTXcore STATIC ${core_includes} ${core_sources} ${core_extern_sources})
target_include_directories(rVTXcore SYSTEM PUBLIC ${core_extern_includes})
target_include_directories(rVTXcore PUBLIC include/)
target_link_libraries(     rVTXcore PUBLIC ${core_extern_libraries})
target_compile_features(   rVTXcore PRIVATE cxx_std_17)
target_compile_options(    rVTXcore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${core_compile_flags}>)
target_compile_definitions(rVTXcore PRIVATE ${core_compile_definition})
add_library(rVTX::core ALIAS rVTXcore)