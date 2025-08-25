# Prepare data for parent scope
set(ui_extern_includes
    ${imgui_includes}
)

set(ui_extern_libraries
    SDL2::SDL2
    SDL2::SDL2main
)

set(ui_extern_sources
    ${imgui_sources}
    ${imgui_sdl_gl_sources}
)

set(ui_extern_includes ${ui_extern_includes} PARENT_SCOPE)
set(ui_extern_libraries ${ui_extern_libraries} PARENT_SCOPE)
set(ui_extern_sources ${ui_extern_sources} PARENT_SCOPE)