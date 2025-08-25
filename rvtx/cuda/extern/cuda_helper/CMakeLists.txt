if(NOT TARGET cuda_helper)
	add_library(cuda_helper INTERFACE)
	target_include_directories(cuda_helper INTERFACE .)
endif()