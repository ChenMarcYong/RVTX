struct VSInput
{
    uint vertexID : SV_VertexID;
};
struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

VSOutput main(VSInput vin)
{
    VSOutput o;
    uint u = vin.vertexID;

    // bits = (0 or 2)
    float2 bits = float2(float((u << 1) & 2u), float(u & 2u));
    o.uv = bits; // UV non utilisées dans le PS "Load"
    o.pos = float4(bits * 2.0f - 1.0f, 0, 1);
    return o;
}
