if(NOT TARGET SDL2::SDL2)
    set(SDL2_DISABLE_UNINSTALL      ON  CACHE BOOL "" FORCE) # Name clash between SDL2 and chemfiles
    set(SDL_TEST                    OFF CACHE BOOL "" FORCE)
    set(SDL_TEST_ENABLED_BY_DEFAULT OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS                   OFF CACHE BOOL "" FORCE)
    add_subdirectory(SDL)
endif()

if(NOT TARGET gl3w)
    add_subdirectory(gl3w)
endif()

if(NOT TARGET tinyexr)
    set(TINYEXR_BUILD_SAMPLE OFF)
    add_subdirectory(tinyexr)
endif()

set(imgui_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imconfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_internal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/misc/cpp/imgui_stdlib.h
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/misc/cpp/imgui_stdlib.cpp
)

set(imgui_sdl_gl_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_sdl2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_sdl2.h
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.h
)

set(imgui_includes
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
)

set(imgui_sources ${imgui_sources} PARENT_SCOPE)
set(imgui_sdl_gl_sources ${imgui_sdl_gl_sources} PARENT_SCOPE)
set(imgui_includes ${imgui_includes} PARENT_SCOPE)

set(tinyexr_includes
    ${CMAKE_CURRENT_SOURCE_DIR}/tinyexr
)

set(tinyexr_includes ${tinyexr_includes} PARENT_SCOPE)

if (RVTX_INCLUDE_CUDA)
    if(NOT TARGET sesdf)
        add_subdirectory(SESDF)
    endif()
endif()


add_subdirectory(diligentEngine)

