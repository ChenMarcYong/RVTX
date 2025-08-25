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


add_dependency_folder(diligent_ball_Data
    "${CMAKE_CURRENT_SOURCE_DIR}/data"          # source (ton dossier avec scene_2AGA.json)
    "${CMAKE_BINARY_DIR}/bin/data"             # destination (à côté de l'exécutable)
)

# L'exécutable dépend de la copie des données
add_dependencies(diligent_pipeline_deffered_ball diligent_ball_Data)