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

    float4 environmentParams0; // x=skyboxExposure, y=skyboxLod, z=iblIntensity, w=showSkybox
    float4 environmentParams1; // x=diffuseIBLIntensity, y=specularIBLIntensity, z=debugReflectionOnly, w=enableIBL
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

[[vk::binding(3, 1)]]
SamplerCube environmentTexture;

[shader("fragment")]
float4 main(SkyboxVSOutput input) : SV_Target
{
    if (ubo.environmentParams0.w < 0.5)
    {
        discard;
    }

    float exposure = ubo.environmentParams0.x;
    float lod = ubo.environmentParams0.y;

    float3 dir = normalize(input.viewDir);

    // Cubemap convention adapter you found working
    dir = float3(dir.x, dir.z, dir.y);

    float3 color = environmentTexture.SampleLevel(dir, lod).rgb;
    color *= exposure;

    return float4(color, 1.0);
}