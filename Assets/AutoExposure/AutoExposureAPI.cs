using System;
using UnityEngine;

public static class AutoExposureAPI
{
    private static ComputeShader findShader(string name)
    {
        return (ComputeShader)Resources.Load(name);
    }

    public static float GetExposureValue(Texture2D input)
    {
        ComputeShader shader = findShader("ComputeExposure");
        if (shader == null)
        {
            Debug.LogError("Could not find exposure compute shader.");
            return 0.0f;
        }

        int kernel = shader.FindKernel("ComputeExposureMain");
        if (kernel < 0)
        {
            Debug.LogError("Could not find ComputeExposureMain kernel.");
            return 0.0f;
        }

        ComputeBuffer rowInfoBuffer = new ComputeBuffer((input.height + 63) / 64, sizeof(float) * 2);
        float[] rowInfos = new float[((input.height + 63) / 64) * 2];

        shader.SetTexture(kernel, "InputTexture", input);
        shader.SetInt("InputTextureWidth", input.width);
        shader.SetBuffer(kernel, "GroupMaxBuffer", rowInfoBuffer);

        // Do work!
        shader.Dispatch(kernel, (input.height + 63) / 64, 1, 1);

        // get sum of groups
        rowInfoBuffer.GetData(rowInfos);

        // find sum of all groups
        float lumSum = 0.0f;
        float nonBlack = 0.0f;
        for (int group = 0; group < (input.height + 63) / 64; group++)
        {
            lumSum += rowInfos[2 * group + 1];
            nonBlack += rowInfos[2 * group + 0];
        }
        float exposureValue = 1.0f;
        if (nonBlack > 0)
        {
            const float key = 0.18f;
            float sum = lumSum / nonBlack;
            exposureValue = key / Mathf.Pow(2.0f, sum);
        }

        // Deallocate
        if (null != rowInfoBuffer)
        {
            rowInfoBuffer.Release();
        }

        return exposureValue;
    }

    public static Texture Map(Texture input, float exposureValue)
    {
        ComputeShader mapShader = findShader("MapHDR");
        if(mapShader == null)
        {
            Debug.LogError("Could not find compute kernel.");
            return null;
        }

        int handleMapHDRMain = mapShader.FindKernel("MapHDRMain");
        if (handleMapHDRMain < 0)
        {
            Debug.LogError("Could not find kernel.");
            return null;
        }

        // Can we write directly to Texture2D instead?
        RenderTexture outputRT = new RenderTexture(input.width, input.height, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
        outputRT.enableRandomWrite = true;
        outputRT.Create();

        mapShader.SetTexture(handleMapHDRMain, "InputTexture", input);
        mapShader.SetTexture(handleMapHDRMain, "OutputTexture", outputRT);
        mapShader.SetFloat("exposure", exposureValue);
        mapShader.Dispatch(handleMapHDRMain, input.width, input.height, 1);

        return outputRT;
    }

    public static Texture Unmap(Texture input, float exposureValue)
    {
        ComputeShader mapShader = findShader("UnmapHDR");
        if (mapShader == null)
        {
            Debug.LogError("Could not find unmap compute kernel.");
            return null;
        }

        int handleUnmapHDRMain = mapShader.FindKernel("UnmapHDRMain");
        if (handleUnmapHDRMain < 0)
        {
            Debug.LogError("Could not find unmap kernel.");
            return null;
        }

        // Can we write directly to Texture2D instead?
        RenderTexture outputRT = new RenderTexture(input.width, input.height, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
        outputRT.enableRandomWrite = true;
        outputRT.Create();

        mapShader.SetTexture(handleUnmapHDRMain, "InputTexture", input);
        mapShader.SetTexture(handleUnmapHDRMain, "OutputTexture", outputRT);
        mapShader.SetFloat("inverseExposure", 1.0f / exposureValue);
        mapShader.Dispatch(handleUnmapHDRMain, input.width, input.height, 1);

        return outputRT;
    }
}
