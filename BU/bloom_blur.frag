[[vk::binding(0, 0)]]
Sampler2D inputTexture;

struct BlurParams
{
    float directionX;
    float directionY;
    float texelSizeX;
    float texelSizeY;
};

[[vk::push_constant]]
ConstantBuffer<BlurParams> pc;

[shader("fragment")]
float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float2 texel = float2(pc.texelSizeX, pc.texelSizeY);
    float2 direction = float2(pc.directionX, pc.directionY);

    float3 result = inputTexture.Sample(uv).rgb * 0.227027;

    result += inputTexture.Sample(uv + direction * texel * 1.384615).rgb * 0.316216;
    result += inputTexture.Sample(uv - direction * texel * 1.384615).rgb * 0.316216;

    result += inputTexture.Sample(uv + direction * texel * 3.230769).rgb * 0.070270;
    result += inputTexture.Sample(uv - direction * texel * 3.230769).rgb * 0.070270;

    return float4(result, 1.0);
}