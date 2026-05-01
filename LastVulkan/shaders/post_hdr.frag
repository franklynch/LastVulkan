[[vk::binding(0, 0)]]
Sampler2D hdrTexture;

struct PostParams
{
    float exposure;
    float toneMappingEnabled;
    float gammaEnabled;
    float padding;
};

[[vk::push_constant]]
ConstantBuffer<PostParams> pc;

float3 ApplyGamma(float3 color)
{
    return pow(color, 1.0 / 2.2);
}

float3 ToneMapACES(float3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[shader("fragment")]
float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float3 color = hdrTexture.Sample(uv).rgb;

    color *= pc.exposure;

    if (pc.toneMappingEnabled > 0.5)
    {
        color = ToneMapACES(color);
    }

    if (pc.gammaEnabled > 0.5)
    {
        color = ApplyGamma(color);
    }

    return float4(color, 1.0);
}