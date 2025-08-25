#pragma once
#include "rvtx/ux/input.hpp"

struct InputAdapter {
    virtual ~InputAdapter() = default;
    virtual void Poll(float dt, rvtx::Input& out) = 0; // remplit 'out'
};
