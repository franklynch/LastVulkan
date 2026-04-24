static const float PI = 3.14159265359;

struct VSOutput
{
    float4 pos : SV_Position;
    float3 localDir : TEXCOORD0;
};

[[vk::binding(0, 0)]]
SamplerCube environmentMap;

[shader("fragment")]
float4 main(VSOutput input) : SV_Target
{
    float3 N = normalize(input.localDir);

    float3 up    = float3(0.0, 1.0, 0.0);
    float3 right = normalize(cross(up, N));
    up           = normalize(cross(N, right));

    float sampleDelta = 0.025;
    float3 irradiance = float3(0.0, 0.0, 0.0);
    float nrSamples = 0.0;

    for (float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for (float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            float3 tangentSample = float3(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
            );

            float3 sampleVec =
                tangentSample.x * right +
                tangentSample.y * up +
                tangentSample.z * N;

            irradiance +=
                environmentMap.Sample(normalize(sampleVec)).rgb *
                cos(theta) * sin(theta);

            nrSamples += 1.0;
        }
    }

    irradiance = PI * irradiance / max(nrSamples, 1.0);

    return float4(irradiance, 1.0);
}