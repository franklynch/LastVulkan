struct VSInput
{
    float3 inPosition : POSITION;
    float3 inNormal   : NORMAL;
    float2 inTexCoord : TEXCOORD0;
    float4 inTangent  : TANGENT;
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

    float4 environmentParams0;       // x=skyboxExposure, y=skyboxLod, z=iblIntensity, w=showSkybox
    float4 environmentParams1;       // x=diffuseIBLIntensity, y=specularIBLIntensity, z=debugReflectionOnly, w=enableIBL
    float4 postProcessParams;        // x=postExposure, y=toneMappingEnabled, z=gammaEnabled, w=environmentRotation
    float4 environmentControlParams; // x=rotateSkybox, y=rotateIBLLighting, z/w unused
    int4 debugParams;
    float4 specularDebugParams;
    float4 specularCurveParams;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

struct PushConstants
{
    float4x4 model;
    float4x4 normalMatrix;
    float4 baseColorFactor;
    float4 materialParams;   // x=metallic, y=roughness, z=normalScale, w=alphaCutoff
    float4 alphaModeParams;  // x=isMask, y=isBlend, z/w unused
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 1)]]
Sampler2D baseColorTexture;

[[vk::binding(1, 1)]]
Sampler2D normalTexture;

[[vk::binding(2, 1)]]
Sampler2D metallicRoughnessTexture;

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
    float4 pos           : SV_Position;
    float2 fragTexCoord  : TEXCOORD0;
    float3 fragNormal    : TEXCOORD1;
    float3 fragTangent   : TEXCOORD2;
    float3 fragBitangent : TEXCOORD3;
    float3 worldPos      : TEXCOORD4;
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    float4 worldPos = mul(pc.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));
    output.worldPos = worldPos.xyz;

    output.fragTexCoord = input.inTexCoord;

    float3x3 model3x3 = (float3x3)pc.model;
    float3x3 normalMatrix = (float3x3)pc.normalMatrix;

    float3 N = normalize(mul(normalMatrix, input.inNormal));
    float3 T = normalize(mul(model3x3, input.inTangent.xyz));

    T = normalize(T - N * dot(N, T));

    float3 B = normalize(cross(N, T) * input.inTangent.w);

    output.fragNormal = N;
    output.fragTangent = T;
    output.fragBitangent = B;

    return output;
}

static const float PI = 3.14159265359;

static const int DEBUG_FINAL = 0;
static const int DEBUG_ALBEDO = 1;
static const int DEBUG_NORMAL = 2;
static const int DEBUG_ROUGHNESS = 3;
static const int DEBUG_METALLIC = 4;
static const int DEBUG_DIRECT_ONLY = 5;
static const int DEBUG_DIFFUSE_IBL_ONLY = 6;
static const int DEBUG_SPECULAR_IBL_ONLY = 7;
static const int DEBUG_REFLECTION_VECTOR = 8;

float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;

    float NdotH = saturate(dot(N, H));
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 0.000001);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    return NdotV / max(NdotV * (1.0 - k) + k, 0.000001);
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));

    float ggxV = GeometrySchlickGGX(NdotV, roughness);
    float ggxL = GeometrySchlickGGX(NdotL, roughness);

    return ggxV * ggxL;
}

float3 ToneMapACES(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    color = (color * (a * color + b)) / (color * (c * color + d) + e);
    return saturate(color);
}

float3 ApplyGamma(float3 color)
{
    return pow(max(color, 0.0), float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

float3 RotateEnvironmentY(float3 dir, float angleRadians)
{
    float s = sin(angleRadians);
    float c = cos(angleRadians);

    return float3(
        c * dir.x + s * dir.z,
        dir.y,
       -s * dir.x + c * dir.z
    );
}

float3 ToEnvironmentDirection(float3 dir)
{
    dir = float3(dir.x, dir.z, dir.y);
    return normalize(dir);
}

[shader("fragment")]
float4 fragMain(VSOutput input, bool isFrontFace : SV_IsFrontFace) : SV_Target
{
    float4 texColor = baseColorTexture.Sample(input.fragTexCoord);
    float4 baseColor = texColor * pc.baseColorFactor;

    bool isMaskMaterial = pc.alphaModeParams.x > 0.5;
    bool isBlendMaterial = pc.alphaModeParams.y > 0.5;

    float alphaCutoff = pc.materialParams.w;

    if (isMaskMaterial && baseColor.a < alphaCutoff)
    {
        discard;
    }

    float normalScale = pc.materialParams.z;

    float3 tangentNormal = normalTexture.Sample(input.fragTexCoord).xyz;
    tangentNormal = tangentNormal * 2.0 - 1.0;
    tangentNormal.xy *= normalScale;
    tangentNormal = normalize(tangentNormal);

    float3 T = normalize(input.fragTangent);
    float3 B = normalize(input.fragBitangent);
    float3 Ng = normalize(input.fragNormal);

    if (!isFrontFace)
    {
        T = -T;
        B = -B;
        Ng = -Ng;
    }

    float3x3 TBN = float3x3(T, B, Ng);
    float3 N = normalize(mul(tangentNormal, TBN));

    float3 mrSample = metallicRoughnessTexture.Sample(input.fragTexCoord).rgb;

    float metallic = saturate(pc.materialParams.x * mrSample.b);
    float roughness = clamp(pc.materialParams.y * mrSample.g, 0.08, 1.0);

    float3 V = normalize(ubo.cameraPosition.xyz - input.worldPos);
    float NdotV = max(dot(N, V), 0.001);

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);

    float skyboxExposure = ubo.environmentParams0.x;
    float skyboxLod = ubo.environmentParams0.y;
    float iblIntensity = ubo.environmentParams0.z;

    float diffuseIBLIntensity = ubo.environmentParams1.x;
    float specularIBLIntensity = ubo.environmentParams1.y;
    bool debugReflectionOnly = ubo.environmentParams1.z > 0.5;
    bool enableIBL = ubo.environmentParams1.w > 0.5;

    float environmentRotation = ubo.postProcessParams.w;
    bool rotateIBLLighting = ubo.environmentControlParams.y > 0.5;

    if (debugReflectionOnly)
    {
        float3 R = reflect(-V, N);
        R = ToEnvironmentDirection(R);

        if (rotateIBLLighting)
        {
            R = RotateEnvironmentY(R, environmentRotation);
        }

        float3 reflection = environmentTexture.SampleLevel(R, skyboxLod).rgb;
        float outputAlpha = isBlendMaterial ? baseColor.a : 1.0;

        return float4(reflection * skyboxExposure, outputAlpha);
    }

    float3 L = normalize(-ubo.lightDirection.xyz);
    float3 H = normalize(V + L);

    float NdotL = saturate(dot(N, L));
    float HdotV = saturate(dot(H, V));

    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    float3 F = FresnelSchlick(HdotV, F0);

    float3 specularDirect =
        (D * G * F) /
        max(4.0 * NdotV * NdotL, 0.001);

    float3 kdDirect = (1.0 - F) * (1.0 - metallic);
    float3 diffuseDirect = kdDirect * baseColor.rgb / PI;

    float3 directLighting =
        (diffuseDirect + specularDirect) *
        ubo.lightColor.rgb *
        NdotL;

    directLighting += baseColor.rgb * ubo.ambientColor.rgb * 0.1;

    float3 diffuseIBL = float3(0.0, 0.0, 0.0);
    float3 specularIBL = float3(0.0, 0.0, 0.0);

    if (enableIBL)
    {
        float3 Fibl = FresnelSchlick(NdotV, F0);
        float3 kdIBL = (1.0 - Fibl) * (1.0 - metallic);

        float3 Nenv = ToEnvironmentDirection(N);

        if (rotateIBLLighting)
        {
            Nenv = RotateEnvironmentY(Nenv, environmentRotation);
        }

        float3 irradiance = irradianceTexture.Sample(Nenv).rgb;
        diffuseIBL = irradiance * baseColor.rgb * kdIBL;

        float3 R = reflect(-V, N);
        R = ToEnvironmentDirection(R);

        if (rotateIBLLighting)
        {
            R = RotateEnvironmentY(R, environmentRotation);
        }

       float maxPrefilterMip = max(ubo.specularDebugParams.z, 0.0);
        float roughnessMipScale = ubo.specularDebugParams.w;
        float roughnessMipBias = ubo.specularCurveParams.x;

        float prefilterLod =
            roughness * maxPrefilterMip * roughnessMipScale
            + roughnessMipBias;

        if (ubo.specularDebugParams.x > 0.5)
        {
            prefilterLod = ubo.specularDebugParams.y;
        }

        prefilterLod = clamp(prefilterLod, 0.0, maxPrefilterMip);

        float3 prefiltered = prefilteredTexture.SampleLevel(R, prefilterLod).rgb;
        float2 brdf = brdfLutTexture.Sample(float2(NdotV, roughness)).rg;

        specularIBL = prefiltered * (Fibl * brdf.x + brdf.y);
    }

    float3 color = directLighting;

    if (enableIBL)
    {
        float3 iblDiffuse = diffuseIBL * diffuseIBLIntensity;
        float3 iblSpecular = specularIBL * specularIBLIntensity;

        color += (iblDiffuse + iblSpecular) * iblIntensity;
    }

    int debugViewMode = ubo.debugParams.x;
    float outputAlpha = isBlendMaterial ? baseColor.a : 1.0;

    if (debugViewMode == DEBUG_ALBEDO)
    {
        return float4(baseColor.rgb, outputAlpha);
    }

    if (debugViewMode == DEBUG_NORMAL)
    {
        return float4(normalize(N) * 0.5 + 0.5, outputAlpha);
    }

    if (debugViewMode == DEBUG_ROUGHNESS)
    {
        return float4(roughness.xxx, outputAlpha);
    }

    if (debugViewMode == DEBUG_METALLIC)
    {
        return float4(metallic.xxx, outputAlpha);
    }

    if (debugViewMode == DEBUG_DIRECT_ONLY)
    {
        return float4(directLighting, outputAlpha);
    }

    if (debugViewMode == DEBUG_DIFFUSE_IBL_ONLY)
    {
        float3 debugDiffuse = enableIBL
            ? diffuseIBL * diffuseIBLIntensity * iblIntensity
            : float3(0.0, 0.0, 0.0);

        return float4(debugDiffuse, outputAlpha);
    }

    if (debugViewMode == DEBUG_SPECULAR_IBL_ONLY)
    {
        float3 debugSpecular = enableIBL
            ? specularIBL * specularIBLIntensity * iblIntensity
            : float3(0.0, 0.0, 0.0);

        return float4(debugSpecular, outputAlpha);
    }

    if (debugViewMode == DEBUG_REFLECTION_VECTOR)
    {
        float3 debugR = reflect(-V, N);
        return float4(normalize(debugR) * 0.5 + 0.5, outputAlpha);
    }

    float postExposure = ubo.postProcessParams.x;
    bool toneMappingEnabled = ubo.postProcessParams.y > 0.5;
    bool gammaEnabled = ubo.postProcessParams.z > 0.5;

    color *= postExposure;

    if (toneMappingEnabled)
    {
        color = ToneMapACES(color);
    }

    if (gammaEnabled)
    {
        color = ApplyGamma(color);
    }

    

    return float4(color, outputAlpha);
}