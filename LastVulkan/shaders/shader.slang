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

    float4 lightDirection;
    float4 lightColor;
    float4 ambientColor;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

struct PushConstants
{
    float4x4 model;
    float4 baseColorFactor;
    float4 materialParams;
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

    // Good enough for now.
    // Later, if you introduce non-uniform scaling heavily,
    // switch to a proper normal matrix.
    output.fragNormal = normalize(mul((float3x3)pc.model, input.inNormal));

    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target
{
    float3 N = normalize(input.fragNormal);
    float3 L = normalize(-ubo.lightDirection.xyz);

    float metallic = saturate(pc.materialParams.x);
    float roughness = saturate(pc.materialParams.y);

    float NdotL = max(dot(N, L), 0.0);
    
    float diffuseStrength = 1.0 - metallic;
    
    float lightSharpness = lerp(1.5, 0.5, roughness);
    
    
    float3 lighting = ubo.ambientColor.xyz + ubo.lightColor.xyz * NdotL * diffuseStrength * lightSharpness;


    float4 texColor = texture.Sample(input.fragTexCoord);
    float4 baseColor = texColor * pc.baseColorFactor;

    return float4(baseColor.rgb * lighting, baseColor.a);
}