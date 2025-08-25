if (RVTX_INCLUDE_CUDA)
	check_language(CUDA)
	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		find_package(CUDAToolkit REQUIRED) # Needed for PTX compilation
	else()
		message("Could NOT find CUDA, disabling features.")
		set(RVTX_INCLUDE_CUDA OFF CACHE BOOL "Includes CUDA library to rVTX project." FORCE)
	endif()

	if (CMAKE_CUDA_COMPILER AND RVTX_INCLUDE_OPTIX)
		find_package(OptiX)
		if(NOT OptiX_INCLUDE)
			message("Could NOT find OptiX, disabling features.")
			set(RVTX_INCLUDE_OPTIX OFF CACHE BOOL "Includes OptiX library to rVTX project." FORCE)
		endif()
	else() 
		set(RVTX_INCLUDE_OPTIX OFF CACHE BOOL "Includes OptiX library to rVTX project." FORCE)
	endif()
else()
	set(RVTX_INCLUDE_CUDA OFF CACHE BOOL "Includes CUDA library to rVTX project." FORCE)
	set(RVTX_INCLUDE_OPTIX OFF CACHE BOOL "Includes OptiX library to rVTX project." FORCE)
endif()

add_subdirectory(extern)
add_subdirectory(core)

if (RVTX_INCLUDE_CUDA)
	add_subdirectory(cuda)
endif()

if (RVTX_INCLUDE_GL)
	add_subdirectory(gl)
endif()


if (RVTX_INCLUDE_DIL)
	add_subdirectory(dil)
endif()


add_subdirectory(raytracing)

if (RVTX_INCLUDE_UI)
	add_subdirectory(ui)
endif()


