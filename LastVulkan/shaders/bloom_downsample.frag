[[vk::binding(0, 0)]]
Sampler2D inputTexture;

struct DownsampleParams
{
    float texelSizeX;
    float texelSizeY;
    float padding0;
    float padding1;
};

[[vk::push_constant]]
ConstantBuffer<DownsampleParams> pc;

[shader("fragment")]
float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float2 texel = float2(pc.texelSizeX, pc.texelSizeY);

    float3 c = float3(0.0, 0.0, 0.0);

    c += inputTexture.Sample(uv + texel * float2(-1.0, -1.0)).rgb;
    c += inputTexture.Sample(uv + texel * float2( 1.0, -1.0)).rgb;
    c += inputTexture.Sample(uv + texel * float2(-1.0,  1.0)).rgb;
    c += inputTexture.Sample(uv + texel * float2( 1.0,  1.0)).rgb;

    c += inputTexture.Sample(uv).rgb * 4.0;

    c /= 8.0;

    return float4(c, 1.0);
}