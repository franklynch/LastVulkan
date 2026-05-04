struct VSOutput
{
    float4 pos : SV_Position;
    float3 localDir : TEXCOORD0;
};

struct PushConstants
{
    float4x4 viewProj;
    float4 params; // x = roughness
};

[[vk::push_constant]]
PushConstants pc;

[shader("vertex")]
VSOutput main(uint vertexID : SV_VertexID)
{
    float3 cubeVertices[36] =
    {
        float3(-1,-1,-1), float3(-1, 1,-1), float3(-1, 1, 1),
        float3(-1,-1,-1), float3(-1, 1, 1), float3(-1,-1, 1),

        float3( 1,-1, 1), float3( 1, 1, 1), float3( 1, 1,-1),
        float3( 1,-1, 1), float3( 1, 1,-1), float3( 1,-1,-1),

        float3(-1, 1, 1), float3(-1, 1,-1), float3( 1, 1,-1),
        float3(-1, 1, 1), float3( 1, 1,-1), float3( 1, 1, 1),

        float3(-1,-1,-1), float3(-1,-1, 1), float3( 1,-1, 1),
        float3(-1,-1,-1), float3( 1,-1, 1), float3( 1,-1,-1),

        float3( 1,-1,-1), float3( 1, 1,-1), float3(-1, 1,-1),
        float3( 1,-1,-1), float3(-1, 1,-1), float3(-1,-1,-1),

        float3(-1,-1, 1), float3(-1, 1, 1), float3( 1, 1, 1),
        float3(-1,-1, 1), float3( 1, 1, 1), float3( 1,-1, 1)
    };

    VSOutput output;

    float3 localPos = cubeVertices[vertexID];
    output.localDir = localPos;

    float4 clip = mul(pc.viewProj, float4(localPos, 1.0));
    output.pos = clip.xyww;

    return output;
}