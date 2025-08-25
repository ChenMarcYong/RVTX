define_cxx_flags(ui_compile_flags ui_compile_definition)

add_subdirectory(extern)

set(ui_includes
)

set(ui_sources
)

set(ui_extern_libraries ${ui_extern_libraries}
    rVTX::core
)

add_library(			   rVTXui STATIC ${ui_includes} ${ui_sources} ${ui_extern_sources})
target_include_directories(rVTXui SYSTEM PUBLIC ${ui_extern_includes})
target_include_directories(rVTXui PUBLIC include/)
target_link_libraries(     rVTXui PUBLIC ${ui_extern_libraries})
target_compile_features(   rVTXui PRIVATE cxx_std_17)
target_compile_options(    rVTXui PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ui_compile_flags}>)
target_compile_definitions(rVTXui PRIVATE ${ui_compile_definition})
add_library(rVTX::ui ALIAS rVTXui)