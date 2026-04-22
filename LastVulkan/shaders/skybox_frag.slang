struct SkyboxVSOutput
{
    float4 pos : SV_Position;
    float3 viewDir : TEXCOORD0;
};

[[vk::binding(3, 1)]]
SamplerCube environmentTexture;

[shader("fragment")]
float4 main(SkyboxVSOutput input) : SV_Target
{
    float3 dir = normalize(input.viewDir);
    dir = float3(dir.x, dir.z, dir.y);
    float3 color = environmentTexture.Sample(dir).rgb;
    return float4(color, 1.0);
}