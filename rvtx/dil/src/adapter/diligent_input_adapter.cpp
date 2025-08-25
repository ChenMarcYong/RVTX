#include "rvtx/dil/adapter/diligent_input_adapter.hpp"

using namespace Diligent;


#ifdef _WIN32
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

static inline bool IsVkDown(int vk)
{
#ifdef _WIN32
    return (::GetAsyncKeyState(vk) & 0x8000) != 0;
#else
    return false;
#endif
}


void DiligentInputAdapter::Poll(float dt, rvtx::Input& m_Input)
{
    // 0) reset des événements "Down/Up/Clicked"
    m_Input.reset();
    //m_Input.keysPressed.clear();
    m_Input.deltaTime = dt;

    // 1) fenêtre
    const auto& sc = m_sc->GetDesc();
    m_Input.windowSize = { sc.Width, sc.Height };
    m_Input.windowResized = m_windowResizedFlag;
    m_windowResizedFlag = false;

    // 2) souris
    const auto& ms = m_ic.GetMouseState();

    m_Input.mousePosition = { int(ms.PosX), int(ms.PosY) };
    m_Input.deltaMousePosition = { int(ms.PosX) - m_prevMouseX, int(ms.PosY) - m_prevMouseY };
    m_prevMouseX = int(ms.PosX);
    m_prevMouseY = int(ms.PosY);

    const bool leftNow = (ms.ButtonFlags & MouseState::BUTTON_FLAG_LEFT) != 0;
    const bool rightNow = (ms.ButtonFlags & MouseState::BUTTON_FLAG_RIGHT) != 0;
    const bool middleNow = (ms.ButtonFlags & MouseState::BUTTON_FLAG_MIDDLE) != 0;

    const bool leftPrev = (m_prevMouseButtons & MouseState::BUTTON_FLAG_LEFT) != 0;
    const bool rightPrev = (m_prevMouseButtons & MouseState::BUTTON_FLAG_RIGHT) != 0;
    const bool middlePrev = (m_prevMouseButtons & MouseState::BUTTON_FLAG_MIDDLE) != 0;

    m_Input.mouseLeftPressed = leftNow;
    m_Input.mouseRightPressed = rightNow;
    m_Input.mouseMiddlePressed = middleNow;

    m_Input.mouseLeftClicked = leftNow && !leftPrev;
    m_Input.mouseRightClicked = rightNow && !rightPrev;
    m_Input.mouseMiddleClicked = middleNow && !middlePrev;

    m_prevMouseButtons = ms.ButtonFlags;

    int wheel = int(ms.WheelDelta);
    if (m_ic.IsKeyDown(InputKeys::ZoomIn))  wheel += 1;
    if (m_ic.IsKeyDown(InputKeys::ZoomOut)) wheel -= 1;
    m_Input.deltaMouseWheel = wheel;

    // 3) clavier : map Diligent::InputKeys -> rvtx::Key, avec détection Up/Down

    auto poll = [&](rvtx::Key rkey, std::initializer_list<Diligent::InputKeys> actions)
        {
            bool now = false;
            for (auto ik : actions) now = now || m_ic.IsKeyDown(ik);

            bool was = m_prevKey.count(rkey) ? m_prevKey[rkey] : false;

            if (now) {
                m_Input.keysPressed.insert(rkey);
                if (!was) m_Input.keysDown.insert(rkey);   // transition Up->Down
            }
            else {
                m_Input.keysPressed.erase(rkey);           // <-- crucial : enlève la touche “collée”
                if (was) m_Input.keysUp.insert(rkey);      // transition Down->Up
            }

            m_prevKey[rkey] = now;
        };




    // ZQSD + flèches + WASD (si tu veux)
    poll(rvtx::Key::Z, { InputKeys::MoveForward });
    poll(rvtx::Key::S, { InputKeys::MoveBackward });
    poll(rvtx::Key::Q, { InputKeys::MoveLeft });
    poll(rvtx::Key::D, { InputKeys::MoveRight });

    poll(rvtx::Key::W, { InputKeys::MoveForward });
    poll(rvtx::Key::A, { InputKeys::MoveLeft });
    poll(rvtx::Key::Left, { InputKeys::MoveLeft });
    poll(rvtx::Key::Right, { InputKeys::MoveRight });

    // Modificateurs
    poll(rvtx::Key::LShift, { InputKeys::ShiftDown });
    poll(rvtx::Key::LCtrl, { InputKeys::ControlDown });
    poll(rvtx::Key::LAlt, { InputKeys::AltDown });


    // map "raw key" -> rvtx::Key avec détection Down/Up/Pressed
    auto poll_vk = [&](rvtx::Key rkey, int vk)
        {
            const bool now = IsVkDown(vk);
            const bool was = m_prevKey[rkey];

            if (now) {
                m_Input.keysPressed.insert(rkey);
                if (!was) m_Input.keysDown.insert(rkey);   // transition Up->Down
            }
            else {
                m_Input.keysPressed.erase(rkey);
                if (was) m_Input.keysUp.insert(rkey);      // transition Down->Up
            }
            m_prevKey[rkey] = now;
        };
    poll_vk(rvtx::Key::F7, VK_F7);
    poll_vk(rvtx::Key::F8, VK_F8);
    poll_vk(rvtx::Key::P, 'P');

}
