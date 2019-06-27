using System;
using System.Collections;
using System.Collections.Generic;
using Barracuda;
using UnityEngine;

// Though not strictly needed, we had to make a namespace to make the C++ -> C# invocation work.
namespace BarracudaDenoiser
{
    static public class API
    {
        private static void DummyTest(Texture2D input, RenderTexture output)
        {
            ComputeShader dummyShader = (ComputeShader)Resources.Load("Dummy");
            if (dummyShader == null)
            {
                Debug.LogError("Could not find compute kernel.");
                return;
            }

            int dummyKernel = dummyShader.FindKernel("DummyMain");
            if (dummyKernel < 0)
            {
                Debug.LogError("Could not find kernel.");
                return;
            }

            dummyShader.SetTexture(dummyKernel, "InputTexture", input);
            dummyShader.SetTexture(dummyKernel, "OutputTexture", output);
            dummyShader.Dispatch(dummyKernel, input.width, input.height, 1);

            //Debug.Log("C# is done!");
        }

        unsafe static public void Denoise(Texture2D input, RenderTexture output)
        {
            DummyTest(input, output);

            //float exposureValue = AutoExposure.GetExposureValue(input);
            //Debug.Log("Exposure: " + exposureValue);
        }
    }
}
