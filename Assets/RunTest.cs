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

	public TextAsset weightsJSON;

	private Model model;
	private IWorker engine;
	private Tensor inputTensor;
	private RenderTexture rt;

	// Use this for initialization
	void Start ()
	{
		Application.targetFrameRate = 60;
		
		model = OIDNModel.BuildModel(weightsJSON, inputImage.height, inputImage.width);
		engine = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.ComputePrecompiled, model, true);
		
		inputTensor = new Tensor(inputImage);

		StartCoroutine(RunInference());
	}
	
	IEnumerator RunInference ()
	{
		// Skip frame before starting
		yield return null;
		displayImage.texture = inputImage;

		Profiler.BeginSample("Schedule execution");
		engine.Execute(inputTensor);
		Profiler.EndSample();


		Profiler.BeginSample("Fetch execution results");
		var output = engine.Peek();
		Profiler.EndSample();

		rt = BarracudaTextureUtils.TensorToRenderTexture(output);
		displayImage.texture = rt;
		
		//output.PrintDataPart(50000);
		output.PrintDataPart(1024);


		engine.Dispose();
		inputTensor.Dispose();
	}
}
