using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Though not strictly needed, we had to make a namespace to make the C++ -> C# invocation work.
namespace BarracudaDenoiser
{
    static public class API
    {
        unsafe static public void Test(IntPtr ptr, int length)
        {
            //a[0] = 123f;
            //Debug.Log("Hello: " + a[1] + ", " + length);
            //Debug.Log(" " + a);
            int* a = (int*)ptr;
            for (int x = 0; x < length; ++x)
            {
                Debug.Log(a[x]);
            }
        }
    }
}
