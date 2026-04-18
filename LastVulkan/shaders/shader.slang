struct VSInput
{
    float3 inPosition;
    float3 inNormal;
    float2 inTexCoord;
    float4 inTangent;
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
    float4 materialParams; // x = metallic, y = roughness, z = normalScale
};

[[vk::push_constant]]
PushConstants pc;

// Set 1, Binding 0 = base color
[[vk::binding(0, 1)]]
Sampler2D baseColorTexture;

// Set 1, Binding 1 = normal
[[vk::binding(1, 1)]]
Sampler2D normalTexture;

[[vk::binding(2, 1)]]
Sampler2D metallicRoughnessTexture;

struct VSOutput
{
    float4 pos : SV_Position;
    float2 fragTexCoord;
    float3 fragNormal;
    float3 fragTangent;
    float3 fragBitangent;
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    float4 worldPos = mul(pc.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));

    output.fragTexCoord = input.inTexCoord;

    float3 N = normalize(mul((float3x3)pc.model, input.inNormal));
    float3 T = normalize(mul((float3x3)pc.model, input.inTangent.xyz));
    float3 B = normalize(cross(N, T) * input.inTangent.w);

    output.fragNormal = N;
    output.fragTangent = T;
    output.fragBitangent = B;

    return output;
}
[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target
{
    float metallic = saturate(pc.materialParams.x);
    float roughness = saturate(pc.materialParams.y);
    float normalScale = pc.materialParams.z;

    float3 tangentNormal = normalTexture.Sample(input.fragTexCoord).xyz * 2.0 - 1.0;
    tangentNormal.xy *= normalScale;
    tangentNormal = normalize(tangentNormal);

    float3x3 TBN = float3x3(
        normalize(input.fragTangent),
        normalize(input.fragBitangent),
        normalize(input.fragNormal));

    float3 N = normalize(mul(TBN, tangentNormal));
    float3 L = normalize(-ubo.lightDirection.xyz);

    float3 mrSample = metallicRoughnessTexture.Sample(input.fragTexCoord).rgb;
    roughness *= mrSample.g;
    metallic *= mrSample.b;

    float NdotL = max(dot(N, L), 0.0);

    float diffuseStrength = 1.0 - metallic;
    float lightSharpness = lerp(1.5, 0.5, roughness);

    float3 lighting =
        ubo.ambientColor.xyz +
        ubo.lightColor.xyz * NdotL * diffuseStrength * lightSharpness;

    float4 texColor = baseColorTexture.Sample(input.fragTexCoord);
    float4 baseColor = texColor * pc.baseColorFactor;

    return float4(baseColor.rgb * lighting, baseColor.a);
}