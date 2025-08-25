#include "rvtx/gl/window.hpp"

#include <GL/gl3w.h>
#include <SDL.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_sdl2.h>

#include "rvtx/core/logger.hpp"

namespace rvtx::gl
{
    static void APIENTRY debugMessageCallback( const GLenum   p_source,
                                               const GLenum   p_type,
                                               const GLuint   p_id,
                                               const GLenum   p_severity,
                                               const GLsizei  p_length,
                                               const GLchar * p_msg,
                                               const void *   p_data )
    {
        if ( p_severity == GL_DEBUG_SEVERITY_NOTIFICATION )
            return;

        std::string source;
        std::string type;
        std::string severity;

        switch ( p_source )
        {
        case GL_DEBUG_SOURCE_API: source = "API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM: source = "WINDOW SYSTEM"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: source = "SHADER COMPILER"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY: source = "THIRD PARTY"; break;
        case GL_DEBUG_SOURCE_APPLICATION: source = "APPLICATION"; break;
        case GL_DEBUG_SOURCE_OTHER: source = "UNKNOWN"; break;
        default: source = "UNKNOWN"; break;
        }

        switch ( p_type )
        {
        case GL_DEBUG_TYPE_ERROR: type = "ERROR"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: type = "DEPRECATED BEHAVIOR"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: type = "UDEFINED BEHAVIOR"; break;
        case GL_DEBUG_TYPE_PORTABILITY: type = "PORTABILITY"; break;
        case GL_DEBUG_TYPE_PERFORMANCE: type = "PERFORMANCE"; break;
        case GL_DEBUG_TYPE_OTHER: type = "OTHER"; break;
        case GL_DEBUG_TYPE_MARKER: type = "MARKER"; break;
        default: type = "UNKNOWN"; break;
        }

        switch ( p_severity )
        {
        case GL_DEBUG_SEVERITY_HIGH: severity = "HIGH"; break;
        case GL_DEBUG_SEVERITY_MEDIUM: severity = "MEDIUM"; break;
        case GL_DEBUG_SEVERITY_LOW: severity = "LOW"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: severity = "NOTIFICATION"; break;
        default: severity = "UNKNOWN"; break;
        }

        rvtx::logger::debug( "[OPENGL] [{}] [{}] {} : {}", severity, type, source, p_msg );
    }

    Window::Window( std::string title, std::size_t width, std::size_t height, bool shown ) :
        m_title( std::move( title ) ), m_width( width ), m_height( height )
    {
        logger::debug( "Initializing SDL2" );

        if ( SDL_Init( SDL_INIT_VIDEO | SDL_INIT_TIMER ) != 0 )
            throw std::runtime_error( SDL_GetError() );

        SDL_GL_SetAttribute( SDL_GL_CONTEXT_FLAGS, 0 );
        SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE );
        SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 4 );
        SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 5 );
        SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );
        SDL_GL_SetAttribute( SDL_GL_DEPTH_SIZE, 24 );
        SDL_GL_SetAttribute( SDL_GL_STENCIL_SIZE, 8 );

        SDL_DisplayMode displayMode;
        SDL_GetCurrentDisplayMode( 0, &displayMode );

        const uint32_t visibilityFlag = shown ? SDL_WINDOW_SHOWN : SDL_WINDOW_HIDDEN;

        m_window
            = SDL_CreateWindow( m_title.c_str(),
                                SDL_WINDOWPOS_CENTERED,
                                SDL_WINDOWPOS_CENTERED,
                                static_cast<int>( width ),
                                static_cast<int>( height ),
                                visibilityFlag | SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI );

        if ( !m_window )
            throw std::runtime_error( SDL_GetError() );

        m_glContext = SDL_GL_CreateContext( m_window );
        if ( !m_glContext )
            throw std::runtime_error( SDL_GetError() );

        SDL_GL_MakeCurrent( m_window, m_glContext );

        logger::debug( "SDL2 initialized" );

        if ( gl3wInit() )
            throw std::runtime_error( "gl3wInit() failed" );

        if ( !gl3wIsSupported( 4, 5 ) )
            throw std::runtime_error( "OpenGL version not supported" );

        rvtx::logger::debug( "Initializing OpenGL" );
        if ( gl3wInit() )
            throw std::runtime_error( "gl3wInit() failed" );

        if ( !gl3wIsSupported( 4, 5 ) )
            throw std::runtime_error( "OpenGL version not supported" );

#ifndef NDEBUG
        glEnable( GL_DEBUG_OUTPUT );
        glDebugMessageCallback( debugMessageCallback, NULL );
#endif // NDEBUG

        logger::debug( "OpenGL initialized" );

        if ( !IMGUI_CHECKVERSION() )
            throw std::runtime_error( "IMGUI_CHECKVERSION() failed" );

        ImGui::CreateContext();

        // Setup controls.
        ImGuiIO & io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        // Style.
        ImGui::StyleColorsDark();
        ImGui::GetStyle().WindowRounding    = 0.f;
        ImGui::GetStyle().ChildRounding     = 0.f;
        ImGui::GetStyle().FrameRounding     = 0.f;
        ImGui::GetStyle().GrabRounding      = 0.f;
        ImGui::GetStyle().PopupRounding     = 0.f;
        ImGui::GetStyle().ScrollbarRounding = 0.f;
        ImGui::GetStyle().WindowBorderSize  = 0.f;
        ImGui::GetStyle().WindowPadding     = ImVec2( 0.f, 0.f );

        // Setup Platform/Renderer bindings.
        if ( ImGui_ImplSDL2_InitForOpenGL( m_window, m_glContext ) == false )
            throw std::runtime_error( "ImGui_ImplSDL2_InitForOpenGL failed" );

        if ( ImGui_ImplOpenGL3_Init( "#version 450" ) == false )
            throw std::runtime_error( "ImGui_ImplOpenGL3_Init failed" );

        rvtx::logger::debug( "ImGui initialized" );
    }

    Window::Window( Window && other ) noexcept
    {
        std::swap( m_title, other.m_title );
        std::swap( m_width, other.m_width );
        std::swap( m_height, other.m_height );
        std::swap( m_window, other.m_window );
        std::swap( m_glContext, other.m_glContext );
        std::swap( m_input, other.m_input );
        std::swap( m_lastTimeStep, other.m_lastTimeStep );
        std::swap( m_isVisible, other.m_isVisible );
    }

    Window & Window::operator=( Window && other ) noexcept
    {
        std::swap( m_title, other.m_title );
        std::swap( m_width, other.m_width );
        std::swap( m_height, other.m_height );
        std::swap( m_window, other.m_window );
        std::swap( m_glContext, other.m_glContext );
        std::swap( m_input, other.m_input );
        std::swap( m_lastTimeStep, other.m_lastTimeStep );
        std::swap( m_isVisible, other.m_isVisible );

        return *this;
    }

    Window::~Window()
    {
        ImGui_ImplSDL2_Shutdown();
        ImGui_ImplOpenGL3_Shutdown();
        if ( ImGui::GetCurrentContext() )
            ImGui::DestroyContext();

        if ( m_glContext )
            SDL_GL_DeleteContext( m_glContext );

        if ( m_window )
            SDL_DestroyWindow( m_window );

        SDL_Quit();
    }

    static Key toKey( SDL_Scancode scanCode );

    bool Window::update()
    {
        m_input.reset();

        // Based on ImGui
        // https://github.com/ocornut/imgui/blob/master/backends/imgui_impl_sdl.cpp#L557
        static const uint64_t sdlFrequency = SDL_GetPerformanceFrequency();
        const uint64_t        now          = SDL_GetPerformanceCounter();
        m_input.deltaTime = static_cast<float>( static_cast<double>( now - m_lastTimeStep ) / sdlFrequency );
        m_lastTimeStep    = now;

        bool      running = true;
        SDL_Event windowEvent;

        ImGuiIO & io = ImGui::GetIO();
        while ( SDL_PollEvent( &windowEvent ) )
        {
            ImGui_ImplSDL2_ProcessEvent( &windowEvent );
            switch ( windowEvent.type )
            {
            case SDL_QUIT: running = false; break;
            case SDL_MOUSEWHEEL:
            {
                m_input.deltaMouseWheel = windowEvent.wheel.y;
                break;
            }
            case SDL_MOUSEMOTION:
            {
                m_input.deltaMousePosition.x = windowEvent.motion.xrel;
                m_input.deltaMousePosition.y = windowEvent.motion.yrel;
                m_input.mousePosition.x      = windowEvent.motion.x;
                m_input.mousePosition.y      = windowEvent.motion.y;
                break;
            }
            case SDL_KEYDOWN:
            {
                Key key = toKey( windowEvent.key.keysym.scancode );
                m_input.keysPressed.emplace( key );
                m_input.keysDown.emplace( key );
                break;
            }
            case SDL_KEYUP:
            {
                Key key = toKey( windowEvent.key.keysym.scancode );
                m_input.keysPressed.erase( key );
                m_input.keysUp.emplace( key );
                break;
            }
            case SDL_MOUSEBUTTONDOWN:
            {
                switch ( windowEvent.button.button )
                {
                case SDL_BUTTON_LEFT:
                {
                    m_input.mouseLeftPressed = true;
                    m_input.mouseLeftClicked = true;
                    m_input.doubleLeftClick  = windowEvent.button.clicks == 2;
                    break;
                }
                case SDL_BUTTON_RIGHT:
                    m_input.mouseRightPressed = true;
                    m_input.mouseRightClicked = true;
                    break;
                case SDL_BUTTON_MIDDLE:
                    m_input.mouseMiddlePressed = true;
                    m_input.mouseMiddleClicked = true;
                    break;
                }
                break;
            }
            case SDL_MOUSEBUTTONUP:
            {
                switch ( windowEvent.button.button )
                {
                case SDL_BUTTON_LEFT: m_input.mouseLeftPressed = false; break;
                case SDL_BUTTON_RIGHT: m_input.mouseRightPressed = false; break;
                case SDL_BUTTON_MIDDLE: m_input.mouseMiddlePressed = false; break;
                }
                break;
            }
            case SDL_WINDOWEVENT:
            {
                if ( windowEvent.window.event == SDL_WINDOWEVENT_SIZE_CHANGED )
                {
                    m_width  = windowEvent.window.data1;
                    m_height = windowEvent.window.data2;

                    m_input.windowSize    = { m_width, m_height };
                    m_input.windowResized = true;
                }

                break;
            }
            case SDL_DROPFILE:
            {
                m_input.droppedFile = std::string( windowEvent.drop.file );
                SDL_free( windowEvent.drop.file );
                break;
            }
            }
        }

        return running;
    }

    void Window::resize( std::size_t width, std::size_t height ) { SDL_SetWindowSize( m_window, width, height ); }

    static Key toKey( SDL_Scancode scanCode )
    {
        switch ( scanCode )
        {
        case SDL_SCANCODE_A: return Key::A;
        case SDL_SCANCODE_B: return Key::B;
        case SDL_SCANCODE_C: return Key::C;
        case SDL_SCANCODE_D: return Key::D;
        case SDL_SCANCODE_E: return Key::E;
        case SDL_SCANCODE_F: return Key::F;
        case SDL_SCANCODE_G: return Key::G;
        case SDL_SCANCODE_H: return Key::H;
        case SDL_SCANCODE_I: return Key::I;
        case SDL_SCANCODE_J: return Key::J;
        case SDL_SCANCODE_K: return Key::K;
        case SDL_SCANCODE_L: return Key::L;
        case SDL_SCANCODE_M: return Key::M;
        case SDL_SCANCODE_N: return Key::N;
        case SDL_SCANCODE_O: return Key::O;
        case SDL_SCANCODE_P: return Key::P;
        case SDL_SCANCODE_Q: return Key::Q;
        case SDL_SCANCODE_R: return Key::R;
        case SDL_SCANCODE_S: return Key::S;
        case SDL_SCANCODE_T: return Key::T;
        case SDL_SCANCODE_U: return Key::U;
        case SDL_SCANCODE_V: return Key::V;
        case SDL_SCANCODE_W: return Key::W;
        case SDL_SCANCODE_X: return Key::X;
        case SDL_SCANCODE_Y: return Key::Y;
        case SDL_SCANCODE_Z: return Key::Z;

        case SDL_SCANCODE_RETURN: return Key::Return;
        case SDL_SCANCODE_ESCAPE: return Key::Escape;
        case SDL_SCANCODE_BACKSPACE: return Key::BackSpace;
        case SDL_SCANCODE_TAB: return Key::Tab;
        case SDL_SCANCODE_SPACE: return Key::Space;

        case SDL_SCANCODE_F1: return Key::F1;
        case SDL_SCANCODE_F2: return Key::F2;
        case SDL_SCANCODE_F3: return Key::F3;
        case SDL_SCANCODE_F4: return Key::F4;
        case SDL_SCANCODE_F5: return Key::F5;
        case SDL_SCANCODE_F6: return Key::F6;
        case SDL_SCANCODE_F7: return Key::F7;
        case SDL_SCANCODE_F8: return Key::F8;
        case SDL_SCANCODE_F9: return Key::F9;
        case SDL_SCANCODE_F10: return Key::F10;
        case SDL_SCANCODE_F11: return Key::F11;
        case SDL_SCANCODE_F12: return Key::F12;

        case SDL_SCANCODE_RIGHT: return Key::Right;
        case SDL_SCANCODE_LEFT: return Key::Left;
        case SDL_SCANCODE_DOWN: return Key::Down;
        case SDL_SCANCODE_UP: return Key::Up;

        case SDL_SCANCODE_LCTRL: return Key::LCtrl;
        case SDL_SCANCODE_LSHIFT: return Key::LShift;
        case SDL_SCANCODE_LALT: return Key::LAlt; /**< alt, option */
        case SDL_SCANCODE_LGUI: return Key::LGui; /**< windows, command (apple), meta */
        case SDL_SCANCODE_RCTRL: return Key::RCtrl;
        case SDL_SCANCODE_RSHIFT: return Key::RShift;
        case SDL_SCANCODE_RALT: return Key::RAlt; /**< alt gr, option */
        case SDL_SCANCODE_RGUI: return Key::RGui; /**< windows, command (apple), meta */
        }

        return Key::Unknown;
    }

} // namespace rvtx::gl
