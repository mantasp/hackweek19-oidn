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
	public Texture barracudaOutputTexture;
	public Texture denoiseTexture;
	public float exposureValue;

	// Use this for initialization
	void Start ()
	{
		Application.targetFrameRate = 60;

		if (inputImage != null)
		{
			Destroy(inputImage);
		}
		inputImage = ReadPFM.Read(pfmBytes);	

		exposureValue = AutoExposureAPI.GetExposureValue(inputImage);
		if(exposureValue == 0.0f)
		{
			Debug.Log("Invalid exposure value.");
			return;
		}
		Texture outputMapped = AutoExposureAPI.Map(inputImage, exposureValue);

		model = OIDNModel.BuildModel(weightsJSON, inputImage.height, inputImage.width);
		engine = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.Compute, model, false);

		inputTensor = new Tensor(outputMapped);

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

		barracudaOutputTexture = new RenderTexture(inputImage.width, inputImage.height, 
			0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
		BarracudaTextureUtils.TensorToRenderTexture(output, barracudaOutputTexture as RenderTexture);
	
		// do stuff with outputMapped
		denoiseTexture = AutoExposureAPI.Unmap(barracudaOutputTexture, exposureValue);
		displayImage.texture = denoiseTexture;
	
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
		else if (Input.GetKey(KeyCode.B)) 
		{
			displayImage.texture = barracudaOutputTexture;
		}
		else if (denoiseTexture != null)
		{
			displayImage.texture = denoiseTexture;
		}
	}
}
