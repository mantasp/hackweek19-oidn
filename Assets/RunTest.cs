using System.Collections;
using System.Collections.Generic;
using Barracuda;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.UI;

public class RunTest : MonoBehaviour
{
	private Texture2D inputImage;

	public RawImage displayImage;

	public TextAsset weightsJSON;
	public TextAsset pfmBytes;

	private Model model;
	private IWorker engine;
	private Tensor inputTensor;
	private RenderTexture rt;

	// Use this for initialization
	void Start ()
	{
		Application.targetFrameRate = 60;

		if (inputImage != null)
		{
			Destroy(inputImage);
		}
		inputImage = ReadPFM.Read(pfmBytes);	

		model = OIDNModel.BuildModel(weightsJSON, inputImage.height, inputImage.width);
		engine = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.Compute, model, true);

		inputTensor = new Tensor(inputImage);

		StartCoroutine(RunInference());
	}
	
	IEnumerator RunInference ()
	{
		// Skip frame before starting
		yield return null;
		displayImage.texture = inputImage;
		displayImage.rectTransform.sizeDelta = new Vector2(inputImage.width, inputImage.height);

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

	void Update()
	{
		if (Input.GetKey(KeyCode.Space))
		{
			displayImage.texture = inputImage;
		}
		else if (rt != null)
		{
			displayImage.texture = rt;
		}
	}
}
