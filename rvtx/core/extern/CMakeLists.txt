if(NOT TARGET chemfiles)
    add_subdirectory(chemfiles)
endif()

if(NOT TARGET glm::glm)
    add_subdirectory(glm)
endif()

if(NOT TARGET EnTT)
    add_subdirectory(entt)
endif()

if (NOT TARGET nlohmann_json::nlohmann_json)
    add_subdirectory(nlohmann_json)
endif()

if (NOT TARGET fastgltf)
    add_subdirectory(fastgltf)
endif()

# Prepare data for parent scope
set(core_extern_includes
    ${CMAKE_CURRENT_SOURCE_DIR}/glm
    ${CMAKE_CURRENT_SOURCE_DIR}/plf_nanotimer
    ${CMAKE_CURRENT_SOURCE_DIR}/chemfiles/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thread-pool/include
    ${CMAKE_CURRENT_SOURCE_DIR}/stb
    ${tinyexr_includes}
    ${imgui_includes}
)

set(core_extern_libraries
    fmt
    glm::glm
    chemfiles
    EnTT
    nlohmann_json::nlohmann_json
    tinyexr
    fastgltf
)

set(core_extern_sources
    ${imgui_sources}
)

set(core_extern_sources ${core_extern_sources} PARENT_SCOPE)
set(core_extern_includes ${core_extern_includes} PARENT_SCOPE)
set(core_extern_libraries ${core_extern_libraries} PARENT_SCOPE)
