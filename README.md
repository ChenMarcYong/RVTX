# rVTX

**rVTX** is a library to help research on molecular systems. It provides toolsets and renderers to quickly get started.

## How to use **rVTX**?
**rVTX** is created as a Windows OS library, with NVIDIA GPUs, other vendors and OS may support the library, but were not tested.

### Prerequisites
When using certain modules (more on that later) you must have [CUDA](https://developer.nvidia.com/cuda-11-8-0-download-archive) (version between `11.6` and `11.8` should work) and [OptiX](https://developer.nvidia.com/designworks/optix/download) (versions `7.5`, below untested and above not working).
The library is written in C++17 and a compiler supporting this version should be installed.

### How to add to your project.
**rVTX** is using [CMake](https://cmake.org/) as a build system, to include the rVTX library to your project, first either add it as a submodule using :

```sh
git submodule add https://gitlab.xlim.fr/rVTX-dev/rVTX.git <DestionationFolder>
git submodule update --init --recursive
```
Or clone the project using :
```sh
git clone --recurse-submodules https://gitlab.xlim.fr/rVTX-dev/rVTX.git
```
Then add it to your `CMakeLists.txt` using `add_subdirectory(rVTX)`, then you are free to add any of the targets describer in the [Project Structure](#project-structure) section in targets in your `target_link_libraries( ... )`.

Don't forget to explicitly define output directories in your project or **rVTX** as a library may not work (look at the top of this [CMakeLists.txt](CMakeLists.txt) for an example).

## Project structure

**rVTX** project (`rvtx/`) is divided into several directories, the following table gives the directory of the modules, the corresponding CMake target and a brief explanation of its features. By default, all CMake options are **ON**.
| Directory | CMake target |CMake option | Description |
|-|-|-|-|
| core | rVTX::core | Always included | The core of the library, contains API-agnostic general classes useful across most applications. |
| cuda | rVTX::cuda | RVTX_INCLUDE_CUDA | CUDA wrapper and utility classes. if `RVTX_INCLUDE_GL` is also defined, adds OpenGL interoperability |
| gl | rVTX::gl | RVTX_INCLUDE_GL | OpenGL rendering backend. If `RVTX_INCLUDE_CUDA` is also defined, adds support for [SESDF](https://github.com/PlathC/SESDF). |
| raytracing | rVTX::raytracing | RVTX_INCLUDE_OPTIX | Raytracing rendering backends, currently only OptiX is supported. |
| ui | rVTX::ui | RVTX_INCLUDE_UI | General UI for the **rVTX** project, uses [ImGui](https://github.com/ocornut/imgui). |
| examples |  | RVTX_INCLUDE_EXAMPLES| Example targets for features of the **rVTX** library. (`OFF` by default) |
