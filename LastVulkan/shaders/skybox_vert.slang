struct SkyboxVSOutput
{
    float4 pos : SV_Position;
    float3 viewDir : TEXCOORD0;
};

struct UniformBuffer
{
    float4x4 view;
    float4x4 proj;
    float4x4 invView;
    float4x4 invProj;
    float4 lightDirection;
    float4 lightColor;
    float4 ambientColor;
    float4 cameraPosition;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

[shader("vertex")]
SkyboxVSOutput main(uint vertexID : SV_VertexID)
{
    SkyboxVSOutput output;

    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };

    float2 ndc = positions[vertexID];
    output.pos = float4(ndc, 1.0, 1.0);

    // Reconstruct a view-space ray from clip space
    float4 clip = float4(ndc, 1.0, 1.0);
    float4 view = mul(ubo.invProj, clip);
    float3 dirVS = normalize(view.xyz / view.w);

    // Rotate into world space, ignore translation
    float3 dirWS = mul((float3x3)ubo.invView, dirVS);

    output.viewDir = dirWS;
    return output;
}