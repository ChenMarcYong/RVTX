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
    uint id = vin.vertexID;
    float2 p = float2(float((id << 1) & 2u), float(id & 2u)); // 0 ou 2
    o.uv = p * 0.5f; // 0..1
    o.pos = float4(p * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0, 1);
    return o;
}
