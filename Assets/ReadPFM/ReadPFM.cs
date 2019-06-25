using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ReadPFM
{
    public static Texture2D Read(TextAsset pfmFile)
    {
        BinaryReader reader = new BinaryReader(new MemoryStream(pfmFile.bytes));
        string id = ReadString(reader);
        int C;
        if (id == "PF")
            C = 3;
        else if (id == "Pf")
            C = 1;
        else
            return null;

        int W = int.Parse(ReadString(reader));
        int H = int.Parse(ReadString(reader));
        float scale = Mathf.Abs(float.Parse(ReadString(reader)));

        Texture2D texture = new Texture2D(W, H, TextureFormat.RGBAHalf, false);

        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                Color color;
                if (C == 3) {
                    color = new Color(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), 1f);
                } else {
                    float g = reader.ReadSingle();
                    color = new Color(g, g, g, 1f);
                }

                texture.SetPixel(w, h, color);
            }
        }
        texture.Apply();
        return texture;
    }

    private static string ReadString(BinaryReader reader)
    {
        string str = "";
        while (true)
        {
            char c = reader.ReadChar();
            if (char.IsWhiteSpace(c)) break;
            str = str + c;
        }
        return str;
    }
}
