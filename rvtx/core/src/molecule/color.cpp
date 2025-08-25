#include "rvtx/molecule/color.hpp"

#include <array>

#include "rvtx/molecule/molecule.hpp"

namespace rvtx
{
    // CPK by http://jmol.sourceforge.net/jscolors/#Jmolcolors
    constexpr std::array<glm::vec3, 119> AtomColors = {
        glm::vec3 { 250, 22, 145 },  // UNKNOWN		= 0,
        glm::vec3 { 255, 255, 255 }, // H			= 1,
        glm::vec3 { 217, 255, 255 }, // HE			= 2,
        glm::vec3 { 204, 128, 255 }, // LI			= 3,
        glm::vec3 { 194, 255, 0 },   // BE			= 4,
        glm::vec3 { 255, 181, 181 }, // B			= 5,
        glm::vec3 { 144, 144, 144 }, // C			= 6,
        glm::vec3 { 48, 80, 248 },   // N			= 7,
        glm::vec3 { 255, 13, 13 },   // O			= 8,
        glm::vec3 { 144, 224, 80 },  // F			= 9,
        glm::vec3 { 179, 227, 245 }, // NE			= 10,
        glm::vec3 { 171, 92, 242 },  // NA			= 11,
        glm::vec3 { 138, 255, 0 },   // MG			= 12,
        glm::vec3 { 191, 166, 166 }, // AL			= 13,
        glm::vec3 { 240, 200, 160 }, // SI			= 14,
        glm::vec3 { 255, 128, 0 },   // P			= 15,
        glm::vec3 { 255, 255, 48 },  // S			= 16,
        glm::vec3 { 31, 240, 31 },   // CL			= 17,
        glm::vec3 { 128, 209, 227 }, // AR			= 18,
        glm::vec3 { 143, 64, 212 },  // K			= 19,
        glm::vec3 { 61, 255, 0 },    // CA			= 20,
        glm::vec3 { 230, 230, 230 }, // SC			= 21,
        glm::vec3 { 191, 194, 199 }, // TI			= 22,
        glm::vec3 { 166, 166, 171 }, // V			= 23,
        glm::vec3 { 138, 153, 199 }, // CR			= 24,
        glm::vec3 { 156, 122, 199 }, // MN			= 25,
        glm::vec3 { 224, 102, 51 },  // FE			= 26,
        glm::vec3 { 240, 144, 160 }, // CO			= 27,
        glm::vec3 { 80, 208, 80 },   // NI			= 28,
        glm::vec3 { 200, 128, 51 },  // CU			= 29,
        glm::vec3 { 125, 128, 176 }, // ZN			= 30,
        glm::vec3 { 194, 143, 143 }, // GA			= 31,
        glm::vec3 { 102, 143, 143 }, // GE			= 32,
        glm::vec3 { 189, 128, 227 }, // AS			= 33,
        glm::vec3 { 255, 161, 0 },   // SE			= 34,
        glm::vec3 { 166, 41, 41 },   // BR			= 35,
        glm::vec3 { 92, 184, 209 },  // KR			= 36,
        glm::vec3 { 112, 46, 176 },  // RB			= 37,
        glm::vec3 { 0, 255, 0 },     // SR			= 38,
        glm::vec3 { 148, 255, 255 }, // Y			= 39,
        glm::vec3 { 148, 224, 224 }, // ZR			= 40,
        glm::vec3 { 115, 194, 201 }, // NB			= 41,
        glm::vec3 { 84, 181, 181 },  // MO			= 42,
        glm::vec3 { 59, 158, 158 },  // TC			= 43,
        glm::vec3 { 36, 143, 143 },  // RU			= 44,
        glm::vec3 { 10, 125, 140 },  // RH			= 45,
        glm::vec3 { 0, 105, 133 },   // PD			= 46,
        glm::vec3 { 192, 192, 192 }, // AG			= 47,
        glm::vec3 { 255, 217, 143 }, // CD			= 48,
        glm::vec3 { 166, 117, 115 }, // IN			= 49,
        glm::vec3 { 102, 128, 128 }, // SN			= 50,
        glm::vec3 { 158, 99, 181 },  // SB			= 51,
        glm::vec3 { 212, 122, 0 },   // TE			= 52,
        glm::vec3 { 148, 0, 148 },   // I			= 53,
        glm::vec3 { 66, 158, 176 },  // XE			= 54,
        glm::vec3 { 87, 23, 143 },   // CS			= 55,
        glm::vec3 { 0, 201, 0 },     // BA			= 56,
        glm::vec3 { 112, 212, 255 }, // LA			= 57,
        glm::vec3 { 255, 255, 199 }, // CE			= 58,
        glm::vec3 { 217, 255, 199 }, // PR			= 59,
        glm::vec3 { 199, 255, 199 }, // ND			= 60,
        glm::vec3 { 163, 255, 199 }, // PM			= 61,
        glm::vec3 { 143, 255, 199 }, // SM			= 62,
        glm::vec3 { 97, 255, 199 },  // EU			= 63,
        glm::vec3 { 69, 255, 199 },  // GD			= 64,
        glm::vec3 { 48, 255, 199 },  // TB			= 65,
        glm::vec3 { 31, 255, 199 },  // DY			= 66,
        glm::vec3 { 0, 255, 156 },   // HO			= 67,
        glm::vec3 { 0, 230, 117 },   // ER			= 68,
        glm::vec3 { 0, 212, 82 },    // TM			= 69,
        glm::vec3 { 0, 191, 56 },    // YB			= 70,
        glm::vec3 { 0, 171, 36 },    // LU			= 71,
        glm::vec3 { 77, 194, 255 },  // HF			= 72,
        glm::vec3 { 77, 166, 255 },  // TA			= 73,
        glm::vec3 { 33, 148, 214 },  // W			= 74,
        glm::vec3 { 38, 125, 171 },  // RE			= 75,
        glm::vec3 { 38, 102, 150 },  // OS			= 76,
        glm::vec3 { 23, 84, 135 },   // IR			= 77,
        glm::vec3 { 208, 208, 224 }, // PT			= 78,
        glm::vec3 { 255, 209, 35 },  // AU			= 79,
        glm::vec3 { 184, 184, 208 }, // HG			= 80,
        glm::vec3 { 166, 84, 77 },   // TL			= 81,
        glm::vec3 { 87, 89, 97 },    // PB			= 82,
        glm::vec3 { 158, 79, 181 },  // BI			= 83,
        glm::vec3 { 171, 92, 0 },    // PO			= 84,
        glm::vec3 { 117, 79, 69 },   // AT			= 85,
        glm::vec3 { 66, 130, 150 },  // RN			= 86,
        glm::vec3 { 66, 0, 102 },    // FR			= 87,
        glm::vec3 { 0, 125, 0 },     // RA			= 88,
        glm::vec3 { 112, 171, 250 }, // AC			= 89,
        glm::vec3 { 0, 186, 255 },   // TH			= 90,
        glm::vec3 { 0, 161, 255 },   // PA			= 91,
        glm::vec3 { 0, 143, 255 },   // U			= 92,
        glm::vec3 { 0, 128, 255 },   // NP			= 93,
        glm::vec3 { 0, 107, 255 },   // PU			= 94,
        glm::vec3 { 84, 92, 242 },   // AM			= 95,
        glm::vec3 { 120, 92, 227 },  // CM			= 96,
        glm::vec3 { 138, 79, 227 },  // BK			= 97,
        glm::vec3 { 161, 54, 212 },  // CF			= 98,
        glm::vec3 { 179, 31, 212 },  // ES			= 99,
        glm::vec3 { 179, 31, 186 },  // FM			= 100,
        glm::vec3 { 179, 13, 16 },   // MD			= 101,
        glm::vec3 { 189, 13, 135 },  // NO			= 102,
        glm::vec3 { 199, 0, 102 },   // LR			= 103,
        glm::vec3 { 204, 0, 89 },    // RF			= 104,
        glm::vec3 { 209, 0, 79 },    // DD			= 105,
        glm::vec3 { 217, 0, 69 },    // SG			= 106,
        glm::vec3 { 224, 0, 56 },    // BHJ			= 107,
        glm::vec3 { 230, 0, 46 },    // HS			= 108,
        glm::vec3 { 235, 0, 38 },    // MT			= 109,
        glm::vec3 { 255, 255, 255 }, // DS			= 110,
        glm::vec3 { 255, 255, 255 }, // RG			= 111,
        glm::vec3 { 255, 255, 255 }, // UUB			= 112,
        glm::vec3 { 255, 255, 255 }, // UUT			= 113,
        glm::vec3 { 255, 255, 255 }, // UUQ			= 114,
        glm::vec3 { 255, 255, 255 }, // UUP			= 115,
        glm::vec3 { 255, 255, 255 }, // UUH			= 116,
        glm::vec3 { 255, 255, 255 }, // UUS			= 117,
        glm::vec3 { 255, 255, 255 }  // UUO			= 118,
    };

    glm::vec3 getAtomColor( const rvtx::Atom & atom )
    {
        return AtomColors[ static_cast<uint8_t>( atom.symbol ) ] / 255.f;
    }

    // CPK by http://jmol.sourceforge.net/jscolors/#Jmolcolors
    constexpr std::array<glm::vec3, 26> ChainColors = {
        glm::vec3 { 192, 208, 255 }, // A, a,
        glm::vec3 { 176, 255, 176 }, // B, b,
        glm::vec3 { 255, 192, 200 }, // C, c,
        glm::vec3 { 255, 255, 128 }, // D, d,
        glm::vec3 { 255, 192, 255 }, // E, e,
        glm::vec3 { 176, 240, 240 }, // F, f,
        glm::vec3 { 255, 208, 112 }, // G, g,
        glm::vec3 { 240, 128, 128 }, // H, h,
        glm::vec3 { 245, 222, 179 }, // I, i,
        glm::vec3 { 0, 191, 255 },   // J, j,
        glm::vec3 { 205, 92, 92 },   // K, k,
        glm::vec3 { 102, 205, 170 }, // L, l,
        glm::vec3 { 154, 205, 50 },  // M, m,
        glm::vec3 { 238, 130, 238 }, // N, n,
        glm::vec3 { 0, 206, 209 },   // O, o,
        glm::vec3 { 0, 255, 127 },   // P, p, 0,
        glm::vec3 { 60, 179, 113 },  // Q, q, 1,
        glm::vec3 { 0, 0, 139 },     // R, r, 2,
        glm::vec3 { 189, 183, 107 }, // S, s, 3,
        glm::vec3 { 0, 100, 0 },     // T, t, 4,
        glm::vec3 { 128, 0, 0 },     // U, u, 5,
        glm::vec3 { 128, 128, 0 },   // V, v, 6,
        glm::vec3 { 128, 0, 128 },   // W, w, 7,
        glm::vec3 { 0, 128, 128 },   // X, x, 8,
        glm::vec3 { 184, 134, 11 },  // Y, y, 9,
        glm::vec3 { 178, 34, 34 },   // Z, z
    };

    constexpr glm::vec3 UnknownChainColor = glm::vec3( 255 );
    glm::vec3           getChainColor( const rvtx::Chain & chain )
    {
        if ( chain.id.empty() )
            return UnknownChainColor / 255.f;

        // chain id should be defined by one char
        const char c = static_cast<char>( std::toupper( static_cast<unsigned char>( chain.id[ 0 ] ) ) );

        const int id = static_cast<int>( c ) - 65; // 65 is A
        if ( id < 0 || id > 26 )
            return UnknownChainColor / 255.f;

        return ChainColors[ id ] / 255.f;
    }
} // namespace rvtx
