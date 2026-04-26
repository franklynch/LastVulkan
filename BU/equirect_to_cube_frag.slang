static const float PI = 3.14159265359;

struct VSOutput
{
    float4 pos : SV_Position;
    float3 localDir : TEXCOORD0;
};

[[vk::binding(0, 0)]]
Sampler2D equirectangularMap;

float2 SampleSphericalMap(float3 v)
{
    float2 uv = float2(atan2(v.z, v.x), asin(v.y));
    uv *= float2(0.15915494309189535, 0.3183098861837907);
    uv += 0.5;
    return uv;
}

[shader("fragment")]
float4 main(VSOutput input) : SV_Target
{
    float3 dir = normalize(input.localDir);
    float2 uv = SampleSphericalMap(dir);

    float3 color = equirectangularMap.Sample(uv).rgb;
    return float4(color, 1.0);
}