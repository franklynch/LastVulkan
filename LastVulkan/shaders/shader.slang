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

    float4 environmentParams0; // x=skyboxExposure, y=skyboxLod, z=iblIntensity, w=showSkybox
    float4 environmentParams1; // x=diffuseIBLIntensity, y=specularIBLIntensity, z=debugReflectionOnly, w=enableIBL
    float4 postProcessParams; // x=postExposure, y=toneMappingEnabled, z=gammaEnabled, w=unused
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

struct PushConstants
{
    float4x4 model;
    float4x4 normalMatrix;
    float4 baseColorFactor;
    float4 materialParams; // x = metallic, y = roughness, z = normalScale, w unused
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
    float2 fragTexCoord : TEXCOORD0;
    float3 fragNormal : TEXCOORD1;
    float3 fragTangent : TEXCOORD2;
    float3 fragBitangent : TEXCOORD3;
    float3 worldPos : TEXCOORD4;
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    float4 worldPos = mul(pc.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));
    output.worldPos = worldPos.xyz;

    output.fragTexCoord = input.inTexCoord;

    float3x3 normalMatrix = (float3x3)pc.normalMatrix;

    output.fragNormal = normalize(mul(normalMatrix, input.inNormal));
    output.fragTangent = normalize(mul(normalMatrix, input.inTangent.xyz));
    output.fragBitangent = normalize(
        cross(output.fragNormal, output.fragTangent) * input.inTangent.w
    );

    return output;
}

static const float PI = 3.14159265359;

float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
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
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);

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
    return pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
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
        normalize(input.fragNormal)
    );

    float3 N = normalize(mul(TBN, tangentNormal));

    if (!isFrontFace)
    {
        N = -N;
    }

    float3 mrSample = metallicRoughnessTexture.Sample(input.fragTexCoord).rgb;

    float roughness = saturate(pc.materialParams.y * mrSample.g);
    float metallic = saturate(pc.materialParams.x * mrSample.b);

    float3 V = normalize(ubo.cameraPosition.xyz - input.worldPos);
    float NdotV = max(dot(N, V), 0.0);

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);

    float exposure = ubo.environmentParams0.x;
    float skyboxLod = ubo.environmentParams0.y;
    float iblIntensity = ubo.environmentParams0.z;

    float diffuseIBLIntensity = ubo.environmentParams1.x;
    float specularIBLIntensity = ubo.environmentParams1.y;
    bool debugReflectionOnly = (ubo.environmentParams1.z > 0.5);
    bool enableIBL = (ubo.environmentParams1.w > 0.5);

    // Reflection-only debug mode
    if (debugReflectionOnly)
    {
        float3 R = reflect(-V, N);
        R = float3(R.x, R.z, R.y);

        float3 reflection = environmentTexture.SampleLevel(normalize(R), skyboxLod).rgb;
        reflection *= exposure;

        return float4(reflection, 1.0);
    }

    // -------------------------
    // Direct lighting (GGX / Cook-Torrance)
    // -------------------------
    float3 L = normalize(-ubo.lightDirection.xyz);
    float3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    float3 F = FresnelSchlick(HdotV, F0);

    float3 numerator = D * G * F;
    float denominator = max(4.0 * NdotV * NdotL, 0.000001);
    float3 specularDirect = numerator / denominator;

    float3 kdDirect = (1.0 - F) * (1.0 - metallic);
    float3 diffuseDirect = kdDirect * baseColor.rgb / PI;

    float3 directLighting =
        (diffuseDirect + specularDirect) *
        ubo.lightColor.rgb *
        NdotL;

    // Optional ambient fallback from your existing light setup
    directLighting += baseColor.rgb * ubo.ambientColor.rgb * 0.15;

    // -------------------------
    // IBL
    // -------------------------
    float3 diffuseIBL = float3(0.0, 0.0, 0.0);
    float3 specularIBL = float3(0.0, 0.0, 0.0);

    if (enableIBL)
    {
        float3 Fibl = FresnelSchlick(NdotV, F0);
        float3 kdIBL = (1.0 - Fibl) * (1.0 - metallic);

        float environmentRotation = ubo.postProcessParams.w;

        float3 Nenv = float3(N.x, N.z, N.y);
        Nenv = RotateEnvironmentY(normalize(Nenv), environmentRotation);

        float3 irradiance = irradianceTexture.Sample(Nenv).rgb;


        diffuseIBL = irradiance * baseColor.rgb * kdIBL;

       

       float3 R = reflect(-V, N);
        R = float3(R.x, R.z, R.y);
        R = RotateEnvironmentY(normalize(R), environmentRotation);

        float maxPrefilterMip = 6.0;
        float prefilterLod = roughness * maxPrefilterMip;
        float3 prefiltered = prefilteredTexture.SampleLevel(R, prefilterLod).rgb;
        
        
        R = RotateEnvironmentY(normalize(R), environmentRotation);

        float3 reflection = environmentTexture.SampleLevel(R, skyboxLod).rgb;
         
             
                
                
        float2 brdf = brdfLutTexture.Sample(float2(NdotV, roughness)).rg;

        specularIBL = prefiltered * (Fibl * brdf.x + brdf.y);
    }

    float3 color = directLighting * 0.2;

    if (enableIBL)
    {
        
        color += diffuseIBL * diffuseIBLIntensity * iblIntensity;
        color += specularIBL * specularIBLIntensity * iblIntensity;
    }

    float postExposure = ubo.postProcessParams.x;
    bool toneMappingEnabled = (ubo.postProcessParams.y > 0.5);
    bool gammaEnabled = (ubo.postProcessParams.z > 0.5);

    color *= postExposure;

    if (toneMappingEnabled)
        {
            color = ToneMapACES(color);
        }

    if (gammaEnabled)
        {
            color = ApplyGamma(color);
        } 

    return float4(color, 1.0);
    
    

   
}