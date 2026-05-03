[[vk::binding(0, 0)]]
Sampler2D hdrTexture;

struct BloomExtractParams
{
    float threshold;
    float knee;
    float padding0;
    float padding1;
};

[[vk::push_constant]]
ConstantBuffer<BloomExtractParams> pc;

[shader("fragment")]
float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float3 color = hdrTexture.Sample(uv).rgb;

    float brightness = max(max(color.r, color.g), color.b);

    float soft = brightness - pc.threshold + pc.knee;
    soft = clamp(soft, 0.0, 2.0 * pc.knee);
    soft = (soft * soft) / max(4.0 * pc.knee, 0.0001);

    float contribution = max(brightness - pc.threshold, soft);
    contribution /= max(brightness, 0.0001);

    return float4(color * contribution, 1.0);
}