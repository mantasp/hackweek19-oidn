using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[ExecuteInEditMode]
public class autoExposure : MonoBehaviour
{
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

    float GetLuminance(float r, float g, float b)
    {
        return 0.212671f * r + 0.715160f * g + 0.072169f * b;
    }

    float calculateAutoExposureValue(Texture2D image)
    {
        const float key = 0.18f;
        const float eps = 1e-8f;
        float sumLum = 0.0f;
        int sumNonBlack = 0;
        for (int y = 0; y < image.height; ++y)
        {
            for (int x = 0; x < image.width; ++x)
            {
                Color pix = image.GetPixel(x, y);
                float r = pix.r;
                float g = pix.g;
                float b = pix.b;
                if (float.IsNaN(r))
                    r = 0.0f;
                if (float.IsNaN(g))
                    g = 0.0f;
                if (float.IsNaN(b))
                    b = 0.0f;

                float L = 0.0f;
                L += GetLuminance(r, g, b);
                if (L > eps)
                {
                    sumLum += Mathf.Log(L, 2.0f);
                    sumNonBlack++;
                }
            }
        }
        float sum = sumLum / (float)sumNonBlack;
        return (sumNonBlack > 0) ? (key / Mathf.Pow(2.0f, sum)) : 1.0f;
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

    void ApplyMapping(Texture2D input, Texture2D output, float exposure)
    {
        for (int y = 0; y < input.height; ++y)
        {
            for (int x = 0; x < input.width; ++x)
            {
                Color pix = input.GetPixel(x, y);
                float r = pix.r;
                float g = pix.g;
                float b = pix.b;

                r *= exposure;
                g *= exposure;
                b *= exposure;

                float rT = Mathf.Pow(Mathf.Log((r + 1.0f), 2.0f) * (1.0f / 16.0f), 1.0f / 2.2f);
                float gT = Mathf.Pow(Mathf.Log((g + 1.0f), 2.0f) * (1.0f / 16.0f), 1.0f / 2.2f);
                float bT = Mathf.Pow(Mathf.Log((b + 1.0f), 2.0f) * (1.0f / 16.0f), 1.0f / 2.2f);
                output.SetPixel(x, y, new Color(rT, gT, bT, pix.a));
            }
        }
        output.Apply();
    }

    void ApplyInverseMapping(Texture2D input, Texture2D output, float exposure)
    {
        float rcpExposure = 1.0f / exposure;
        for (int y = 0; y < input.height; ++y)
        {
            for (int x = 0; x < input.width; ++x)
            {
                Color pix = input.GetPixel(x, y);
                float r = pix.r;
                float g = pix.g;
                float b = pix.b;

                float rT = (Mathf.Pow(2.0f, (Mathf.Pow(r, 2.2f) * 16.0f)) - 1.0f) * rcpExposure;
                float gT = (Mathf.Pow(2.0f, (Mathf.Pow(g, 2.2f) * 16.0f)) - 1.0f) * rcpExposure;
                float bT = (Mathf.Pow(2.0f, (Mathf.Pow(b, 2.2f) * 16.0f)) - 1.0f) * rcpExposure;
                output.SetPixel(x, y, new Color(rT, gT, bT, pix.a));
            }
        }
        output.Apply();
    }

    // Update is called once per frame
    void Update()
    {
        if (clear)
        {
            clear = false;
            MakeBlack(outputMapped);
            MakeBlack(outputMappedInverse);
        }
        if (run)
        {
            run = false;
            MakeBlack(outputMapped);
            MakeBlack(outputMappedInverse);
            exposureValue = calculateAutoExposureValue(input);
            ApplyMapping(input, outputMapped, exposureValue);
            ApplyInverseMapping(outputMapped, outputMappedInverse, exposureValue);
        }
    }
}
