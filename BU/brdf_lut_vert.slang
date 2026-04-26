struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

[shader("vertex")]
VSOutput main(uint vertexID : SV_VertexID)
{
    VSOutput output;

    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };

    float2 ndc = positions[vertexID];

    output.pos = float4(ndc, 0.0, 1.0);
    output.uv = ndc * 0.5 + 0.5;

    return output;
}