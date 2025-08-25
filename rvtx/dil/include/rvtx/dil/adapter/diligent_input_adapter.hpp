#pragma once
#include "input_adapter.hpp"
#include "InputController.hpp"
#include "Graphics/GraphicsEngine/interface/SwapChain.h"
#include <unordered_map>

class DiligentInputAdapter : public InputAdapter {
public:
    DiligentInputAdapter(Diligent::InputController& ic, Diligent::ISwapChain* sc)
        : m_ic{ ic }, m_sc{ sc } {
    }

    // Appelle �a depuis ton callback de resize
    void OnResize() { m_windowResizedFlag = true; }

    void Poll(float dt, rvtx::Input& out) override;

private:
    Diligent::InputController& m_ic;
    Diligent::ISwapChain* m_sc = nullptr;

    int m_prevMouseX = 0, m_prevMouseY = 0;
    uint32_t m_prevMouseButtons = 0;
    bool m_windowResizedFlag = false;

    // �tat pr�c�dent de chaque rvtx::Key pour d�tecter Up/Down
    std::unordered_map<rvtx::Key, bool> m_prevKey;
};
