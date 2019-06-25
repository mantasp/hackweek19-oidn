using System.Collections.Generic;
using Barracuda;
using UnityEngine;

class OIDNModel
{
	#pragma warning disable CS0649
	[System.Serializable]
	public class JSONTestSet
	{
		public JSONTensor[] inputs;
		// public JSONTensor[] outputs;
		public JSONTensor GetInputByName(string name) 
		{
			foreach (var input in inputs)
			{
				if (input.name == name) return input;
			}
			return null;
		}
	}

	[System.Serializable]
	public class JSONTensor
	{
		public string name;
		public int[] shape;
		public string type;
		public float[] data;
	}
	#pragma warning restore CS0649


	public static Model BuildModel(TextAsset weightsJSON, int height, int width)
	{
		Model res = new Model();
		// var json = System.File.ReadAllText(fullpath);
		JSONTestSet testSet = JsonUtility.FromJson<JSONTestSet>(weightsJSON.text);

		res.inputs = new Model.Input [] {
			new Model.Input { name = "input" , shape = new int[] { -1, height, width, 3 } }
		};

		List<Layer> layers = new List<Layer>();

		Layer conv1 = AddConvolution(layers, "conv1", false, true, testSet, res.inputs[0].name);
		Layer conv1b = AddConvolution(layers, "conv1b", false, true, testSet, conv1.name);
		Layer pool1 = AddPool(layers, "pool1", conv1b.name);
		Layer conv2 = AddConvolution(layers, "conv2", false, true, testSet, pool1.name);
		Layer pool2 = AddPool(layers, "pool2", conv2.name);
		Layer conv3 = AddConvolution(layers, "conv3", false, true, testSet, pool2.name);
		Layer pool3 = AddPool(layers, "pool3", conv3.name);	
		Layer conv4 = AddConvolution(layers, "conv4", false, true, testSet, pool3.name);
		Layer pool4 = AddPool(layers, "pool4", conv4.name);	
		Layer conv5 = AddConvolution(layers, "conv5", false, true, testSet, pool4.name);
		Layer pool5 = AddPool(layers, "pool5", conv5.name);

		Layer upsample4 = AddUpsample(layers, "upsample4", pool5.name);
		Layer concat4 = AddConcatonate(layers, "concat4", upsample4.name, pool4.name);
		Layer conv6 = AddConvolution(layers, "conv6", false, true, testSet, concat4.name);
		Layer conv6b = AddConvolution(layers, "conv6b", false, true, testSet, conv6.name);

		Layer upsample3 = AddUpsample(layers, "upsample3", conv6b.name);
		Layer concat3 = AddConcatonate(layers, "concat3", upsample3.name, pool3.name);
		Layer conv7 = AddConvolution(layers, "conv7", false, true, testSet, concat3.name);
		Layer conv7b = AddConvolution(layers, "conv7b", false, true, testSet, conv7.name);

		Layer upsample2 = AddUpsample(layers, "upsample2", conv7b.name);
		Layer concat2 = AddConcatonate(layers, "concat2", upsample2.name, pool2.name);
		Layer conv8 = AddConvolution(layers, "conv8", false, true, testSet, concat2.name);
		Layer conv8b = AddConvolution(layers, "conv8b", false, true, testSet, conv8.name);

		Layer upsample1 = AddUpsample(layers, "upsample1", conv8b.name);
		Layer concat1 = AddConcatonate(layers, "concat1", upsample1.name, pool1.name);
		Layer conv9 = AddConvolution(layers, "conv9", false, true, testSet, concat1.name);
		Layer conv9b = AddConvolution(layers, "conv9b", false, true, testSet, conv9.name);

		Layer upsample0 = AddUpsample(layers, "upsample0", conv9b.name);
		Layer concat0 = AddConcatonate(layers, "concat0", upsample0.name, res.inputs[0].name);
		Layer conv10 = AddConvolution(layers, "conv10", false, true, testSet, concat0.name);
		Layer conv10b = AddConvolution(layers, "conv10b", false, true, testSet, conv10.name);

		Layer conv11 = AddConvolution(layers, "conv11", false, false, testSet, conv10b.name);

		res.layers = layers.ToArray();
		res.memories = new Model.Memory[0];

		res.outputs = new string[] { conv11.name };

		return res;
	}

	

	// shape is read from file as Height, Width, Channels, Kernels 
	public static Layer AddConvolution(List<Layer> layers, string name, bool isTranspose, bool addRelu, JSONTestSet testSet, string inputName)
	{
		Layer layer = new Layer();

		JSONTensor weightTensor = testSet.GetInputByName(name + "/W");
		JSONTensor biasTensor = testSet.GetInputByName(name + "/b");
		if (weightTensor == null || biasTensor == null)
		{
			Debug.LogError("weight or bias with base name " + name + " could not be found");
			return null;
		}

		layer.type = isTranspose ? Layer.Type.Conv2DTrans : Layer.Type.Conv2D;
		layer.activation = Layer.Activation.None;
		layer.name = name;
		layer.datasets = new Layer.DataSet[2];
		layer.weights = new float[weightTensor.data.Length + biasTensor.data.Length];
		// copy and concatonate the source weights and biases into the single layer.weights
		System.Array.Copy(weightTensor.data, layer.weights, weightTensor.data.Length);
		int biasOffset = weightTensor.data.Length;
		System.Array.Copy(biasTensor.data, 0, layer.weights, biasOffset, biasTensor.data.Length);
		
		layer.datasets[0].shape = new TensorShape(weightTensor.shape);
		layer.datasets[0].offset = 0;
		layer.datasets[0].length = weightTensor.data.Length;

		int kernels = biasTensor.shape[biasTensor.shape.Length - 1];
		layer.datasets[1].shape = new TensorShape(1, 1, 1, kernels);
		layer.datasets[1].offset = biasOffset;
		layer.datasets[1].length = biasTensor.data.Length;

		layer.stride = new[] {1, 1};
		layer.pad = new[] {1, 1, 1, 1};
		
		layer.inputs  = new [] { inputName };
		layers.Add(layer);

		if (addRelu)
		{
			layer = AddRelu(layers, name + "Relu", layer.name);
		}
		return layer;
	}

	public static Layer AddRelu(List<Layer> layers, string name, string inputName)
	{
		Layer layer = new Layer();
		layer.type = Layer.Type.Activation;
		layer.activation = Layer.Activation.Relu;
		layer.name = name;
		layer.datasets = new Layer.DataSet[0];
		layer.weights = new float[0];
		layer.inputs = new [] {inputName};
		layers.Add(layer);
		return layer;
	}

	public static Layer AddPool(List<Layer> layers, string name, string inputName)
	{
		Layer layer = new Layer();
		layer.type = Layer.Type.MaxPool2D;
		layer.activation = Layer.Activation.None;
		layer.name = name;
		layer.datasets = new Layer.DataSet[0];
		layer.weights = new float[0];
		layer.pool = new [] {2, 2};
		layer.stride = new[] {2, 2};
		// TODO: check the proper pad in pooling
		layer.pad = new[] {0, 0, 0, 0};
		layer.inputs = new [] {inputName};
		layers.Add(layer);
		return layer;
	}

	public static Layer AddUpsample(List<Layer> layers, string name, string inputName)
	{
		Layer layer = new Layer();
		layer.type = Layer.Type.Upsample2D;
		layer.activation = Layer.Activation.None;
		layer.name = name;
		layer.datasets = new Layer.DataSet[0];
		layer.weights = new float[0];
		layer.pool = new [] {2, 2};
		layer.inputs = new [] {inputName};
		layers.Add(layer);
		return layer;
	}

	public static Layer AddConcatonate(List<Layer> layers, string name, string inputA, string inputB)
	{
		Layer layer = new Layer();
		layer.type = Layer.Type.Concat;
		layer.activation = Layer.Activation.None;
		layer.name = name;
		layer.datasets = new Layer.DataSet[0];
		layer.weights = new float[0];
		layer.inputs = new [] {inputA, inputB};
		layer.axis = -1;
		layers.Add(layer);
		return layer;
	}
}