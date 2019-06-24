using System.Collections;
using System.Collections.Generic;
using Barracuda;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.UI;

public class RunTest : MonoBehaviour
{
	public Texture2D inputImage;

	public RawImage displayImage;

	private Model model;
	private IWorker engine;
	private Tensor input;

	// Use this for initialization
	void Start ()
	{
		Application.targetFrameRate = 60;
		
		model = BuildModel(inputImage.height, inputImage.width);
		engine = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.ComputePrecompiled, model, false);
		
		input = new Tensor(inputImage);

		StartCoroutine(RunInference());
	}
	
	IEnumerator RunInference ()
	{
		// Skip frame before starting
		yield return null;
		displayImage.texture = inputImage;

		Profiler.BeginSample("Schedule execution");
		engine.Execute(input);
		Profiler.EndSample();


		Profiler.BeginSample("Fetch execution results");
		var output = engine.Peek();
		Profiler.EndSample();
		
		//output.PrintDataPart(50000);
		output.PrintDataPart(1024);


		engine.Dispose();
		input.Dispose();
	}

	Model BuildModel(int h, int w)
	{
		Model res = new Model();
		
		res.inputs = new Model.Input [] {
			new Model.Input { name = "input" ,       shape = new int[] { -1, h, w, 3 } }
		};
		
		Layer conv1 = new Layer();
		
		var W = new TensorShape(3, 3, 3, 32);
		var B = new TensorShape(1, 1, 1, 32);
		
		conv1.type = Layer.Type.Conv2D;
		conv1.activation = Layer.Activation.None;
		conv1.name = "conv1";
		conv1.datasets = new Layer.DataSet[2];
		conv1.weights = new float[W.length + B.length]; // TODO: fill in
		
		conv1.datasets[0].shape = W;
		conv1.datasets[0].offset = 0;
		conv1.datasets[0].length = W.length;

		conv1.datasets[1].shape = B;
		conv1.datasets[1].offset = W.length;
		conv1.datasets[1].length = B.length;

		conv1.stride = new[] {1, 1};
		conv1.pad = new[] {1, 1, 1, 1};
		
		conv1.inputs  = new [] { res.inputs[0].name };

		res.layers = new[] {conv1};
		res.memories = new Model.Memory[0];

		res.outputs = new string[] { "conv1"};

		return res;
	}
}
