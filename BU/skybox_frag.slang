struct SkyboxVSOutput
{
    float4 pos : SV_Position;
    float3 viewDir : TEXCOORD0;
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
    float4 environmentControlParams; // x=rotateSkybox, y=rotateIBLLighting, z/w unused
    int4 debugParams;
    float4 specularDebugParams;
    float4 specularCurveParams;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

[[vk::binding(3, 1)]]
SamplerCube environmentTexture;

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
float4 main(SkyboxVSOutput input) : SV_Target
{
    if (ubo.environmentParams0.w < 0.5)
    {
        discard;
    }

    float exposure = ubo.environmentParams0.x;
    float lod = ubo.environmentParams0.y;

    float3 dir = normalize(input.viewDir);
    dir = float3(dir.x, dir.z, dir.y);

    float environmentRotation = ubo.postProcessParams.w;
    bool rotateSkybox = (ubo.environmentControlParams.x > 0.5);

    if (rotateSkybox)
    {
        dir = RotateEnvironmentY(dir, environmentRotation);
    }

    float3 color = environmentTexture.SampleLevel(dir, lod).rgb;

    float postExposure = ubo.postProcessParams.x;
    bool toneMappingEnabled = (ubo.postProcessParams.y > 0.5);
    bool gammaEnabled = (ubo.postProcessParams.z > 0.5);

    color *= exposure;
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