struct VSInput
{
    float3 inPosition;
    float3 inNormal;
    float2 inTexCoord;
};

struct UniformBuffer
{
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

struct PushConstants
{
    float4x4 model;
};

[[vk::push_constant]]
PushConstants pc;

// Set 1, Binding 0
[[vk::binding(0, 1)]]
Sampler2D texture;

struct VSOutput
{
    float4 pos : SV_Position;
    float2 fragTexCoord;
    float3 fragNormal;
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    float4 worldPos = mul(pc.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));

    output.fragTexCoord = input.inTexCoord;
    output.fragNormal = input.inNormal;

    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target
{
    return texture.Sample(input.fragTexCoord);
}