# Prepare data for parent scope
set(rt_optix_extern_includes
    ${OptiX_INCLUDE}
)

set(rt_optix_extern_libraries
    sesdf
)

set(rt_optix_extern_sources
)

set(rt_optix_extern_includes ${rt_optix_extern_includes} PARENT_SCOPE)
set(rt_optix_extern_libraries ${rt_optix_extern_libraries} PARENT_SCOPE)
set(rt_optix_extern_sources ${rt_optix_extern_sources} PARENT_SCOPE)