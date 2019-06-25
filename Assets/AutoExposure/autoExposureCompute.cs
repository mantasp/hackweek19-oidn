using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class autoExposureCompute : MonoBehaviour
{
    public ComputeShader computeExposureShader;

    private float lumSum = 0.0f;
    private int nonBlack = 0;
    private float[] groupMaxData;
    private ComputeBuffer groupMaxBuffer;
    private int handleComputeExposureMain;

    public bool run = false;
    public bool clear = false;
    public Texture2D input;
    public float exposureValue;
    //public Texture2D outputMapped;
    //public Texture2D outputMappedInverse;

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
            //MakeBlack(outputMapped);
            //MakeBlack(outputMappedInverse);
            lumSum = 0.0f;
            nonBlack = 0;
            exposureValue = 0.0f;
            clear = false;
        }
        if (run)
        {
            //MakeBlack(outputMapped);
            //MakeBlack(outputMappedInverse);

            if (null == computeExposureShader || null == input/* || null == outputMapped || null == outputMappedInverse*/)
            {
                Debug.Log("Shader or textures missing.");
                run = false;
                return;
            }

            // Allocate and bind
            handleComputeExposureMain = computeExposureShader.FindKernel("ComputeExposureMain");
            groupMaxBuffer = new ComputeBuffer((input.height + 63) / 64, sizeof(float) * 2);
            groupMaxData = new float[((input.height + 63) / 64) * 2];

            if (handleComputeExposureMain < 0 || null == groupMaxBuffer || null == groupMaxData)
            {
                Debug.Log("Initialization failed.");
                run = false;
                return;
            }

            computeExposureShader.SetTexture(handleComputeExposureMain, "InputTexture", input);
            computeExposureShader.SetInt("InputTextureWidth", input.width);
            computeExposureShader.SetBuffer(handleComputeExposureMain, "GroupMaxBuffer", groupMaxBuffer);

            // Do work!
            computeExposureShader.Dispatch(handleComputeExposureMain, (input.height + 63) / 64, 1, 1);
            // divided by 64 in x because of [numthreads(64,1,1)] in the compute shader code
            // added 63 to make sure that there is a group for all rows

            // get maxima of groups
            groupMaxBuffer.GetData(groupMaxData);

            // find maximum of all groups
            lumSum = 0.0f;
            nonBlack = 0;
            for (int group = 0; group < (input.height + 63) / 64; group++)
            {
                lumSum += groupMaxData[2 * group + 1];
                nonBlack += (int)groupMaxData[2 * group + 0];
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

            run = false;
        }
    }
}
