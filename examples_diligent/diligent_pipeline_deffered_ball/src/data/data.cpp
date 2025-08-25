using namespace Diligent;

#include "MapHelper.hpp"
#include "GraphicsUtilities.h"
#include "ColorConversion.h"


constexpr rvtx::dil::Sphere Spheres[] =
{
    {float3{3.f, 0.f, 0.f}, 0.3f, float3{0.f, 1.f, 0.f}, 1.f},
    {float3{0.f, 0.f, 0.f}, 0.5f, float3{1.f, 1.f, 0.f}, 1.f},
    {float3{0.f, 0.f, 5.f}, 1.f,  float3{1.f, 0.f, 1.f}, 1.f},
    {float3{0.f, 0.f, 10.f}, 1.f, float3{1.f, 0.f, 1.f}, 1.f},
    {float3{0.f, 0.f, 20.f}, 1.f, float3{1.f, 0.f, 1.f}, 1.f},
    {float3{0.f, 0.f, 40.f}, 10.f, float3{1.f, 0.8f, 0.5f}, 1.f},
    {float3{0.f, 50.f, 50.f}, 15.f, float3{1.f, 1.f, 0.f}, 1.f},
};


constexpr rvtx::dil::Sphere Spheres2[] =
{
    {float3{-20.f, 0.f, 0.f}, 0.3f, float3{0.f, 1.f, 0.f}, 1.f},
    {float3{-10.f, 0.f, 0.f}, 0.5f, float3{1.f, 1.f, 0.f}, 1.f},
};

constexpr rvtx::dil::Sphere SpheresRed[] =
{
    {float3{20.f, 0.f, 0.f}, 0.3f, float3{1.f, 0.f, 0.f}, 1.f},
    {float3{10.f, 0.f, 0.f}, 0.5f, float3{1.f, 0.f, 0.f}, 1.f},
};

constexpr rvtx::dil::Sphere SpheresBlue[] =
{
    {float3{-20.f, 0.f, 0.f}, 0.3f, float3{0.f, 0.f, 1.f}, 1.f},
    {float3{-10.f, 0.f, 0.f}, 0.5f, float3{0.f, 0.f, 1.f}, 1.f},
};