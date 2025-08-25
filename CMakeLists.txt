cmake_minimum_required(VERSION 3.20)
project(rVTX C CXX)
#fesfe

# Force binaries location for all target of the project if rVTX is not a submodule
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
	
	if(WIN32)
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/bin")
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/lib")
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib")
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/lib")
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib")
	endif()
endif()

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
	"${CMAKE_CURRENT_SOURCE_DIR}/CMake"
	${CMAKE_MODULE_PATH}
)

include(CheckLanguage)
include(CMake/auto_cuda_archicture.cmake)
include(CMake/compile_to_ptx.cmake)
include(CMake/compiler_flags.cmake)
include(CMake/dependency_folder.cmake)

option(RVTX_INCLUDE_CUDA              "Includes CUDA library to the rVTX project."       ON)
option(RVTX_INCLUDE_OPTIX             "Includes OptiX library to the rVTX project."      ON)
option(RVTX_INCLUDE_GL                "Includes OpenGL library to the rVTX project."     ON)
option(RVTX_INCLUDE_UI                "Includes the UI library to the rVTX project."     ON)
option(RVTX_INCLUDE_BINDINGS_PYTHON   "Includes the Python bindings to the rVTX project" OFF)
option(RVTX_INCLUDE_EXAMPLES          "Includes the examples of the rVTX project."       OFF)

#Marc

option(RVTX_INCLUDE_DILIGENT_EXAMPLES          "Includes the examples of Diligent work."       ON)
option(RVTX_INCLUDE_DIL               "Includes DiligentEngine library to the rVTX project."     ON)
option(RVTX_INCLUDE_DILIGENT          "Includes the library DiligentEngine."       ON)


add_subdirectory(rvtx)

if (RVTX_INCLUDE_EXAMPLES)
	add_subdirectory(examples)
endif()


add_subdirectory(examples_diligent)

if (RVTX_INCLUDE_BINDINGS_PYTHON)
	find_package(Python 3.8
		REQUIRED COMPONENTS Interpreter Development.Module
		OPTIONAL_COMPONENTS Development.SABIModule)
	add_subdirectory(bindings/python)
endif()

