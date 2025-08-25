#ifndef RVTX_GL_UTILS_PROGRAM_HPP
#define RVTX_GL_UTILS_PROGRAM_HPP

#include <unordered_map>
#include <vector>

#include "rvtx/core/filesystem.hpp"
#include "rvtx/gl/core/fwd.hpp"

namespace rvtx::gl
{
    class Program
    {
      public:
        Program( std::vector<std::filesystem::path> shaderPaths );

        Program( const Program & )             = delete;
        Program & operator=( const Program & ) = delete;

        Program( Program && ) noexcept;
        Program & operator=( Program && ) noexcept;

        ~Program();

        GLuint getId() const;
        void   setId( GLuint id );
        void   use() const;

        void create( const std::string & name );
        void attachShader( const GLuint shaderId ) const;
        void link();
        void detachShaders();

        const std::vector<std::filesystem::path> & getShaderPaths() const;

      private:
        std::string getProgramErrors() const;

        GLuint                             m_id = GL_INVALID_INDEX;
        std::string                        m_name {};
        std::vector<std::filesystem::path> m_shaderPaths {};

        friend class ProgramManager;
    };

    class ProgramManager
    {
      public:
        ProgramManager( std::string programRoot = {} );

        ProgramManager( const ProgramManager & )             = delete;
        ProgramManager & operator=( const ProgramManager & ) = delete;

        ProgramManager( ProgramManager && ) noexcept;
        ProgramManager & operator=( ProgramManager && ) noexcept;

        ~ProgramManager();

        Program * create( const std::string & name, const std::vector<std::filesystem::path> & paths );
        Program * get( const std::string & );

        void refresh();

        GLuint getShader( std::size_t ) const;

      private:
        GLuint createShader( const std::filesystem::path & path );

        std::string m_programRoot {};

        std::unordered_map<std::string, std::unique_ptr<Program>> m_programs {};
        std::unordered_map<std::size_t, GLuint>                   m_shaders {};
    };
} // namespace rvtx::gl

#endif // RVTX_GL_UTILS_PROGRAM_HPP
