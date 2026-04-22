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

    float4x4 invView;
    float4x4 invProj;

    float4 lightDirection;
    float4 lightColor;
    float4 ambientColor;
    float4 cameraPosition;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

struct PushConstants
{
    float4x4 model;
    float4x4 normalMatrix;
    float4 baseColorFactor;
    float4 materialParams; // x = metallic, y = roughness, z = normalScale
    
    
    
    
};

[[vk::push_constant]]
PushConstants pc;

// ==========================
// Set 1 — Material textures
// ==========================

[[vk::binding(0, 1)]]
Sampler2D baseColorTexture;

[[vk::binding(1, 1)]]
Sampler2D normalTexture;

[[vk::binding(2, 1)]]
Sampler2D metallicRoughnessTexture;


// ==========================
// Set 2 — IBL textures
// ==========================

[[vk::binding(0, 2)]]
SamplerCube irradianceTexture;

[[vk::binding(1, 2)]]
SamplerCube prefilteredTexture;

[[vk::binding(2, 2)]]
Sampler2D brdfLutTexture;

[[vk::binding(3, 2)]]
SamplerCube environmentTexture;

struct VSOutput
{
    float4 pos : SV_Position;
    float2 fragTexCoord;
    float3 fragNormal;
    float3 fragTangent;
    float3 fragBitangent;
    float3 worldPos;
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    float4 worldPos = mul(pc.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));
    output.worldPos = worldPos.xyz;

    output.fragTexCoord = input.inTexCoord;

    // Better normal matrix for world-space transform
    float3x3 normalMatrix = (float3x3)pc.normalMatrix;

    output.fragNormal = normalize(mul(normalMatrix, input.inNormal));
    output.fragTangent = normalize(mul(normalMatrix, input.inTangent.xyz));
    output.fragBitangent = normalize(
        cross(output.fragNormal, output.fragTangent) * input.inTangent.w
    );

    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input, bool isFrontFace : SV_IsFrontFace) : SV_Target
{
    float4 texColor = baseColorTexture.Sample(input.fragTexCoord);
    float4 baseColor = texColor * pc.baseColorFactor;

    float normalScale = pc.materialParams.z;

    float3 tangentNormal = normalTexture.Sample(input.fragTexCoord).xyz * 2.0 - 1.0;
    tangentNormal.xy *= normalScale;
    tangentNormal = normalize(tangentNormal);

    float3x3 TBN = float3x3(
        normalize(input.fragTangent),
        normalize(input.fragBitangent),
        normalize(input.fragNormal));

    float3 N = normalize(mul(TBN, tangentNormal));

    if (!isFrontFace)
        N = -N;

    float3 L = normalize(-ubo.lightDirection.xyz);
    float NdotL = max(dot(N, L), 0.0);

    //  FIX: sample MR texture
    float3 mrSample = metallicRoughnessTexture.Sample(input.fragTexCoord).rgb;

    float roughness = pc.materialParams.y * mrSample.g;
    float metallic  = pc.materialParams.x * mrSample.b;

    float specPower = lerp(64.0, 4.0, roughness);

    float3 specColor = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);

    float3 V = normalize(ubo.cameraPosition.xyz - input.worldPos);
    float3 H = normalize(V + L);

    float spec = pow(max(dot(N, H), 0.0), specPower);

    float3 color =
        baseColor.rgb * (0.3 + 1.2 * NdotL) +
        specColor * spec * NdotL * 0.5;

    return float4(color, 1.0);

}