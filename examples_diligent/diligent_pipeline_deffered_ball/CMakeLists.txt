cmake_minimum_required (VERSION 3.10)

project(diligent_pipeline_deffered_ball CXX)

add_sample_app(diligent_pipeline_deffered_ball
    IDE_FOLDER
        DiligentSamples/Tutorials
    SOURCES
        src/diligent_pipeline_deffered_ball.cpp
    INCLUDES
        src/diligent_pipeline_deffered_ball.hpp
)

target_link_libraries(diligent_pipeline_deffered_ball
    PRIVATE
        rVTX::dil
)