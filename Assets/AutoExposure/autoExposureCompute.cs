using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[ExecuteInEditMode]
public class autoExposureCompute : MonoBehaviour
{
    public ComputeShader computeExposureShader;
    public ComputeShader mapHDRShader;
    public ComputeShader unmapHDRShader;

    private float lumSum = 0.0f;
    private float nonBlack = 0.0f;
    public  float[] groupMaxData;
    private ComputeBuffer groupMaxBuffer;
    private int handleComputeExposureMain;
    private int handleMapHDRMain;
    private int handleUnmapHDRMain;

    public bool run = false;
    public bool clear = false;
    public Texture2D input;
    public float exposureValue;
    public Texture2D outputMapped;
    public Texture2D outputMappedInverse;


    // Start is called before the first frame update
    void Start()
    {
    }

    void MakeBlack(Texture2D inputOutput)
    {
        for (int y = 0; y < inputOutput.height; ++y)
        {
            for (int x = 0; x < inputOutput.width; ++x)
            {
                Color pix = inputOutput.GetPixel(x, y);
                inputOutput.SetPixel(x, y, new Color(0.0f, 0.0f, 0.0f, pix.a));
            }
        }
        inputOutput.Apply();
    }

    // Update is called once per frame
    void Update()
    {
        if (clear)
        {
            if (outputMapped)
                MakeBlack(outputMapped);
            if (outputMappedInverse)
                MakeBlack(outputMappedInverse);
            lumSum = 0.0f;
            nonBlack = 0.0f;
            exposureValue = 0.0f;
            clear = false;
        }
        if (run)
        {
            run = false;
            if (outputMapped)
                MakeBlack(outputMapped);
            if (outputMappedInverse)
                MakeBlack(outputMappedInverse);

            // Check that all the stuff is available
            if (null == computeExposureShader|| null == mapHDRShader || null == unmapHDRShader || null == input || null == outputMapped || null == outputMappedInverse)
            {
                Debug.Log("Shader or textures missing.");
                return;
            }

            // Allocate and bind
            handleComputeExposureMain = computeExposureShader.FindKernel("ComputeExposureMain");
            groupMaxBuffer = new ComputeBuffer((input.height + 63) / 64, sizeof(float) * 2);
            groupMaxData = new float[((input.height + 63) / 64) * 2];

            if (handleComputeExposureMain < 0 || null == groupMaxBuffer || null == groupMaxData)
            {
                Debug.Log("Initialization failed.");
                return;
            }

            computeExposureShader.SetTexture(handleComputeExposureMain, "InputTexture", input);
            computeExposureShader.SetInt("InputTextureWidth", input.width);
            computeExposureShader.SetBuffer(handleComputeExposureMain, "GroupMaxBuffer", groupMaxBuffer);

            // Do work!
            computeExposureShader.Dispatch(handleComputeExposureMain, (input.height + 63) / 64, 1, 1);
            // divided by 64 in x because of [numthreads(64,1,1)] in the compute shader code
            // added 63 to make sure that there is a group for all rows

            // get sum of groups
            groupMaxBuffer.GetData(groupMaxData);

            // find sum of all groups
            lumSum = 0.0f;
            nonBlack = 0.0f;
            for (int group = 0; group < (input.height + 63) / 64; group++)
            {
                lumSum += groupMaxData[2 * group + 1];
                nonBlack += groupMaxData[2 * group + 0];
            }
            exposureValue = 1.0f;
            if (nonBlack > 0)
            {
                const float key = 0.18f;
                float sum = lumSum / nonBlack;
                exposureValue = key / Mathf.Pow(2.0f, sum);
            }
            
            // Deallocate
            if (null != groupMaxBuffer)
            {
                groupMaxBuffer.Release();
            }

            // Allocate a render texture
            RenderTexture rt;
            rt = new RenderTexture(input.width, input.height, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            rt.enableRandomWrite = true;
            rt.Create();

            // Map to HDR
            handleMapHDRMain = mapHDRShader.FindKernel("MapHDRMain");
            if (handleMapHDRMain < 0)
            {
                Debug.Log("Initialization failed.");
                rt.Release();
                return;
            }

            mapHDRShader.SetTexture(handleMapHDRMain, "InputTexture", input);
            mapHDRShader.SetTexture(handleMapHDRMain, "OutputTexture", rt);
            mapHDRShader.SetFloat("exposure", exposureValue);
            mapHDRShader.Dispatch(handleMapHDRMain, input.width, input.height, 1);

            RenderTexture.active = rt;
            outputMapped.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            outputMapped.Apply();
            RenderTexture.active = null;

            // Unmap from HDR
            handleUnmapHDRMain = unmapHDRShader.FindKernel("UnmapHDRMain");
            if (handleUnmapHDRMain < 0)
            {
                Debug.Log("Initialization failed.");
                rt.Release();
                return;
            }

            unmapHDRShader.SetTexture(handleUnmapHDRMain, "InputTexture", outputMapped);
            unmapHDRShader.SetTexture(handleUnmapHDRMain, "OutputTexture", rt);
            unmapHDRShader.SetFloat("inverseExposure", 1.0f/exposureValue);
            unmapHDRShader.Dispatch(handleUnmapHDRMain, input.width, input.height, 1);

            RenderTexture.active = rt;
            outputMappedInverse.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            outputMappedInverse.Apply();
            RenderTexture.active = null;

            rt.Release();
        }
    }
}
