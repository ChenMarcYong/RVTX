struct VSInput
{
    uint vertexID : SV_VertexID; // gl_VertexID
};

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

VSOutput main(VSInput vin)
{
    VSOutput vout;

    // uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uint u = vin.vertexID;
    float2 uvBits = float2(float((u << 1) & 2u), float(u & 2u));
    vout.uv = uvBits;

    // gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    vout.pos = float4(vout.uv * 2.0f - 1.0f, 0.0f, 1.0f);
    
    return vout;
}
