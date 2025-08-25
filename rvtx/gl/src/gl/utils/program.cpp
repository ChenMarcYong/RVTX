#include "rvtx/gl/utils/program.hpp"

#include <GL/gl3w.h>
#include <rvtx/core/logger.hpp>

namespace rvtx::gl
{
    Program::Program( std::vector<std::filesystem::path> shaderPaths ) : m_shaderPaths( std::move( shaderPaths ) ) {}

    Program::Program( Program && other ) noexcept
    {
        std::swap( m_id, other.m_id );
        std::swap( m_name, other.m_name );
        std::swap( m_shaderPaths, other.m_shaderPaths );
    }

    Program & Program::operator=( Program && other ) noexcept
    {
        std::swap( m_id, other.m_id );
        std::swap( m_name, other.m_name );
        std::swap( m_shaderPaths, other.m_shaderPaths );
        return *this;
    }

    Program::~Program()
    {
        if ( glIsProgram( m_id ) )
        {
            // Detach but don't delete shaders, can be used by other programs.
            detachShaders();
            glDeleteProgram( m_id );
        }
    }

    GLuint Program::getId() const { return m_id; }
    void   Program::setId( const GLuint p_id ) { m_id = p_id; }
    void   Program::use() const { glUseProgram( m_id ); }

    void Program::create( const std::string & name )
    {
        if ( glIsProgram( m_id ) )
        {
            rvtx::logger::warning( "Program already created" );
            return;
        }

        m_name = name;
        m_id   = glCreateProgram();
    }

    void Program::attachShader( const GLuint shaderId ) const
    {
        if ( !glIsProgram( m_id ) )
        {
            rvtx::logger::warning( "Cannot attach shader: program is not created" );
            return;
        }

        glAttachShader( m_id, shaderId );
    }

    void Program::link()
    {
        rvtx::logger::debug( "Linking program {}", m_name );

        if ( !glIsProgram( m_id ) )
        {
            rvtx::logger::error( "Can not link program: {} is not created", m_name );
            return;
        }

        GLint linked;
        glLinkProgram( m_id );
        glGetProgramiv( m_id, GL_LINK_STATUS, &linked );
        if ( linked == GL_FALSE )
        {
            glDeleteProgram( m_id );
            rvtx::logger::error( "Error linking program: {}", m_name );
            rvtx::logger::error( getProgramErrors() );
            return;
        }
    }

    void Program::detachShaders()
    {
        GLint nbShaders = 0;
        glGetProgramiv( m_id, GL_ATTACHED_SHADERS, &nbShaders );

        std::vector<GLuint> shaders {};
        shaders.resize( nbShaders );
        glGetAttachedShaders( m_id, nbShaders, nullptr, shaders.data() );
        for ( GLuint shader : shaders )
            glDetachShader( m_id, shader );
    }

    std::string Program::getProgramErrors() const
    {
        GLint length;
        glGetProgramiv( m_id, GL_INFO_LOG_LENGTH, &length );
        if ( length == 0 )
            return "";

        std::vector<GLchar> log;
        log.resize( length );

        glGetProgramInfoLog( m_id, length, &length, &log[ 0 ] );
        return { log.begin(), log.end() };
    }

    const std::vector<std::filesystem::path> & Program::getShaderPaths() const { return m_shaderPaths; }

    enum class ShaderType
    {
        Vertex                 = GL_VERTEX_SHADER,
        Fragment               = GL_FRAGMENT_SHADER,
        Geometry               = GL_GEOMETRY_SHADER,
        Compute                = GL_COMPUTE_SHADER,
        TessellationEvaluation = GL_TESS_EVALUATION_SHADER,
        TessellationControl    = GL_TESS_CONTROL_SHADER,
        Invalid                = GL_INVALID_VALUE
    };

    static const std::unordered_map<std::string_view, ShaderType> ShaderExtensions
        = { { ".vert", ShaderType::Vertex },
            { ".geom", ShaderType::Geometry },
            { ".frag", ShaderType::Fragment },
            { ".comp", ShaderType::Compute },
            { ".tesc", ShaderType::TessellationControl },
            { ".tese", ShaderType::TessellationEvaluation } };

    ShaderType toShaderType( const std::filesystem::path & shaderPath )
    {
        std::string extension = shaderPath.extension().string();
        if ( ShaderExtensions.find( extension ) != ShaderExtensions.end() )
            return ShaderExtensions.at( extension );

        rvtx::logger::warning( "Invalid extension: " + extension );
        return ShaderType::Invalid;
    }

    std::string getShaderErrors( const GLuint shader )
    {
        GLint length;
        glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &length );
        if ( length == 0 )
            return "";

        std::vector<GLchar> log( length );
        glGetShaderInfoLog( shader, length, &length, &log[ 0 ] );
        return { log.begin(), log.end() };
    }

    ProgramManager::ProgramManager( std::string programRoot ) : m_programRoot( std::move( programRoot ) ) {}

    ProgramManager::ProgramManager( ProgramManager && other ) noexcept
    {
        std::swap( m_programRoot, other.m_programRoot );
        std::swap( m_programs, other.m_programs );
        std::swap( m_shaders, other.m_shaders );
    }

    ProgramManager & ProgramManager::operator=( ProgramManager && other ) noexcept
    {
        std::swap( m_programRoot, other.m_programRoot );
        std::swap( m_programs, other.m_programs );
        std::swap( m_shaders, other.m_shaders );

        return *this;
    }

    ProgramManager::~ProgramManager()
    {
        for ( const auto & [ _, handle ] : m_shaders )
            glDeleteShader( handle );
    }

    Program * ProgramManager::create( const std::string & name, const std::vector<std::filesystem::path> & paths )
    {
        rvtx::logger::debug( "Creating program: {}", name );
        if ( m_programs.find( name ) != m_programs.end() )
        {
            rvtx::logger::debug( "Program {} already exists !: ", name );
            return m_programs[ name ].get();
        }

        m_programs[ name ] = std::make_unique<Program>( paths );
        Program & program  = *m_programs[ name ];
        program.create( name );

        for ( const std::filesystem::path & shader : paths )
        {
            const GLuint id = createShader( shader );
            if ( id != GL_INVALID_INDEX )
                program.attachShader( id );
        }

        program.link();

        rvtx::logger::debug( "Program {} created: {}", name, m_programs[ name ]->getId() );
        return m_programs[ name ].get();
    }

    Program * ProgramManager::get( const std::string & name )
    {
        if ( m_programs.find( name ) != m_programs.end() )
            return m_programs.at( name ).get();

        throw std::runtime_error( fmt::format( "Program {} does not exists", name ) );
    }

    void ProgramManager::refresh()
    {
        for ( auto & [ _, program ] : m_programs )
            program->detachShaders();

        // Delete shaders.
        for ( const auto & [ _, handle ] : m_shaders )
            glDeleteShader( handle );

        m_shaders.clear();

        // Then recreate them.
        for ( auto & [ _, program ] : m_programs )
        {
            glDeleteProgram( program->getId() );
            program->setId( glCreateProgram() );
            for ( const std::filesystem::path & shader : program->getShaderPaths() )
            {
                GLuint id = createShader( shader );
                if ( id != GL_INVALID_INDEX )
                    program->attachShader( id );
            }

            program->link();
        }
    }

    GLuint ProgramManager::getShader( const std::size_t p_hash ) const
    {
        if ( m_shaders.find( p_hash ) != m_shaders.end() )
            return m_shaders.at( p_hash );

        return GL_INVALID_INDEX;
    }

    GLuint ProgramManager::createShader( const std::filesystem::path & path )
    {
        rvtx::logger::debug( "Creating shader: {}", path.filename().string() );

        const std::string name = path.string();
        const std::size_t hash = std::hash<std::string> {}( name );

        const ShaderType type = toShaderType( name );
        if ( type == ShaderType::Invalid )
        {
            rvtx::logger::error( "Invalid shader extension: {}", name );
            return GL_INVALID_INDEX;
        }

        GLuint shaderId = getShader( hash );
        if ( shaderId != GL_INVALID_INDEX )
        {
            rvtx::logger::debug( "Shader already exists" );
            return shaderId;
        }

        shaderId                       = glCreateShader( (int)type );
        std::filesystem::path fullPath = m_programRoot / path;
        const std::string     src      = rvtx::read( fullPath );
        if ( src.empty() )
        {
            glDeleteShader( shaderId );
            return GL_INVALID_INDEX;
        }

        const GLchar * shaderCode = src.c_str();
        glShaderSource( shaderId, 1, &shaderCode, 0 );
        glCompileShader( shaderId );
        GLint compiled;
        glGetShaderiv( shaderId, GL_COMPILE_STATUS, &compiled );
        if ( compiled == GL_FALSE )
        {
            glDeleteShader( shaderId );
            rvtx::logger::error( "Error compiling shader: {}", name );
            rvtx::logger::error( "{}", getShaderErrors( shaderId ) );

            return GL_INVALID_INDEX;
        }

        m_shaders.emplace( hash, shaderId );
        rvtx::logger::debug( "Shader created: {}", name );
        return shaderId;
    }

} // namespace rvtx::gl
