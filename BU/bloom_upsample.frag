[[vk::binding(0, 0)]]
Sampler2D inputTexture;

struct UpsampleParams
{
    float intensity;
    float radius;
    float texelSizeX;
    float texelSizeY;
};;

[[vk::push_constant]]
ConstantBuffer<UpsampleParams> pc;

[shader("fragment")]
float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float2 texel = float2(pc.texelSizeX, pc.texelSizeY) * pc.radius;

    float3 sum = float3(0.0, 0.0, 0.0);

    sum += inputTexture.Sample(uv + texel * float2(-1.0, -1.0)).rgb;
    sum += inputTexture.Sample(uv + texel * float2( 1.0, -1.0)).rgb;
    sum += inputTexture.Sample(uv + texel * float2(-1.0,  1.0)).rgb;
    sum += inputTexture.Sample(uv + texel * float2( 1.0,  1.0)).rgb;

    sum += inputTexture.Sample(uv + texel * float2(-1.0,  0.0)).rgb * 2.0;
    sum += inputTexture.Sample(uv + texel * float2( 1.0,  0.0)).rgb * 2.0;
    sum += inputTexture.Sample(uv + texel * float2( 0.0, -1.0)).rgb * 2.0;
    sum += inputTexture.Sample(uv + texel * float2( 0.0,  1.0)).rgb * 2.0;

    sum += inputTexture.Sample(uv).rgb * 4.0;

    sum /= 16.0;

    return float4(sum * pc.intensity, 1.0);
}

