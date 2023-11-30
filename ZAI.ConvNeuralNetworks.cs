using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ZUtilLib.ZAI.FFNeuralNetworks;

namespace ZUtilLib.ZAI.ConvNeuralNetworks
{
	public class ConvolutionalNeuralNetworkMono
	{
		private PolyMatrixInputNodeMono[] InputNodes { get; set; }
		/// <summary>[Layer][Node]</summary>
		private FilterPoolNodeMono[][] ConvAndPoolNodes { get; set; }
		private NeuralNetwork FullyConnectedNN { get; set; }

		private Equations.GraphEquation[] _filterActivFuncs;
		private Operations.ConvOp[] _poolingMethods;
		private int[] _kernelWHs, _poolSampleWHs;
		private (int W, int H) _inputChannelsSize;
		private int _inputNodeCount;
		private bool _initialized = false;

		public ConvolutionalNeuralNetworkMono(int inputNodeCount, int[] kernelWHs, int[] poolSampleWHs, int[] convAndPoolLayerHeights, NDNodeActivFunc[] convActivationFuncs, ConvPoolingOp[] poolingOperations, (int W, int H) inputNodeChannelsSize) // TODO add step settings??? (no pool step though (is auto))
		{
			// Settings verification
			IEnumerable<Array> lengths = new List<Array>() { poolSampleWHs, convAndPoolLayerHeights, convActivationFuncs, poolingOperations }; // It works...
			if (!lengths.All(a => a.Length == kernelWHs.Length))
				throw new Exception("ConvolutionalNeuralNetworkMono critical error: inconsistency with inputted layer setting array lengths.");

			// Fields
			_kernelWHs = kernelWHs;
			_poolSampleWHs = poolSampleWHs;
			_inputNodeCount = inputNodeCount;
			_filterActivFuncs = convActivationFuncs.Select(t => Equations.GetEquationFromType(t)).ToArray();
			_poolingMethods = poolingOperations.Select(t => Operations.GetOperationFromType(t)).ToArray();
			_inputChannelsSize = inputNodeChannelsSize;

			// Array init
			InputNodes = new PolyMatrixInputNodeMono[inputNodeCount];
			ConvAndPoolNodes = new FilterPoolNodeMono[convAndPoolLayerHeights.Length][];
			for (int i = 0; i < convAndPoolLayerHeights.Length; i++)
				ConvAndPoolNodes[i] = new FilterPoolNodeMono[convAndPoolLayerHeights[i]];
		}

		public void InitializeThis(float initialWeightAmp = 1)
		{
			Random random = new Random();
			float GetRandVal() => (float)random.NextDouble() * (initialWeightAmp * 2) - initialWeightAmp;

			// Generate input nodes
			for (int n = 0; n < _inputNodeCount; n++)
				InputNodes[n] = new PolyMatrixInputNodeMono();

			// Generate convolutional & pooling nodes
			for (int layer = 0; layer < ConvAndPoolNodes.Length; layer++)
			{
				for (int n = 0; n < ConvAndPoolNodes[layer].Length; n++)
				{
					// Setup links, randomise associated kernels, and normalize them
					(IMonoConvNeuralNode, float[,])[] nodeLinkKernels;
					if (layer == 0)
					{
						nodeLinkKernels = new (IMonoConvNeuralNode, float[,])[InputNodes.Length];
						for (int inpNode = 0; inpNode < nodeLinkKernels.Length; inpNode++)
						{
							nodeLinkKernels[inpNode] = (
								InputNodes[inpNode],
								((float[,])((ZMatrix)random.NextMatrix(_kernelWHs[layer], _kernelWHs[layer], true) * initialWeightAmp)).NormalizeMatrix(true)
								);
						}
					}
					else
					{
						nodeLinkKernels = new (IMonoConvNeuralNode, float[,])[ConvAndPoolNodes[layer - 1].Length];
						for (int prevLyrNode = 0; prevLyrNode < nodeLinkKernels.Length; prevLyrNode++)
						{
							nodeLinkKernels[prevLyrNode] = (
								ConvAndPoolNodes[layer - 1][prevLyrNode],
								((float[,])((ZMatrix)random.NextMatrix(_kernelWHs[layer], _kernelWHs[layer], true) * initialWeightAmp)).NormalizeMatrix(true)
								);
						}
					}

					ConvAndPoolNodes[layer][n] = new FilterPoolNodeMono(nodeLinkKernels, GetRandVal(), _poolSampleWHs[layer], _filterActivFuncs[layer], _poolingMethods[layer]);
				}
			}

			// Calculate grand total of pixels remaining after all convolutions and pooling for number of inputs
			//(int W, int H)[] layerDimensions = _inputChannelSizes.Clone() as (int W, int H)[];
			//for (int cpLayer = 0; cpLayer < ConvAndPoolNodes.Length; cpLayer++)
			//{
			//	for (int n = 0; n < ConvAndPoolNodes[cpLayer].Length; n++)
			//	{
			//		for (int inLayer = 0; inLayer < layerDimensions.Length; inLayer++)
			//		{
			//			// Conv reduction
			//			layerDimensions[inLayer].W -= _kernelWHs[cpLayer] - 1;
			//			layerDimensions[inLayer].H -= _kernelWHs[cpLayer] - 1;
			//			// Pool reduction
			//			layerDimensions[inLayer].W /= _poolSampleWHs[cpLayer];
			//			layerDimensions[inLayer].H /= _poolSampleWHs[cpLayer];
			//		}
			//	}
			//}
			//int sumOfRemainingPixels = layerDimensions.Sum(d => d.W * d.H);

			_initialized = true;
		}

		public void InitializeThis(ConvolutionalNeuralNetworkMono basedOnNet, float mutateChance, float learningRate)
		{
			// TODO InitializeThis method that is based on the provided network

			// Verify other network specifications, but don't bother too much
			if (basedOnNet == null || !basedOnNet._initialized || basedOnNet.InputNodes.Length != InputNodes.Length || basedOnNet._kernelWHs.Length != _kernelWHs.Length || basedOnNet.ConvAndPoolNodes.Length != ConvAndPoolNodes.Length)
				throw new Exception("InitializeThis critical error: basedOnNet differs from the current network's specs in some way.");

			// CONINUE HERE with the InitializeThis method by adding the carrying over plus potential mutation (research how it should be done for kernels)

			_initialized = true;
		}

		public ConvNetworkResult ComputeResultMono(params float[][,] monoPixelChannels)
		{
			if (!_initialized)
				throw new Exception("ComputeResultMono critical error: CNN not initialized.");
			// Check input validity
			if (monoPixelChannels.Length != _inputNodeCount)
				throw new Exception("ComputeResultMono critical error: invalid number of provided input channels");

			// Normalize all channels on all inputs
			for (int i = 0; i < monoPixelChannels.Length; i++)
				monoPixelChannels[i] = monoPixelChannels[i].NormalizeMatrix(false);

			// TODO Complete computation result method.
			return default;
		}
	}

	internal class FilterPoolNodeMono : IMonoConvNeuralNode
	{
		private float[,] _cData = null;
		public (IMonoConvNeuralNode Node, float[,] Kernel)[] NodeLinkKernels { get; private set; }
		public float Bias { get; private set; }

		private Equations.GraphEquation _activationFunc;
		private Operations.ConvOp _poolOperation;
		private int _poolSampleWH;

		public FilterPoolNodeMono((IMonoConvNeuralNode Node, float[,] Kernel)[] nodeLinkKernels, float bias, int poolSampleWH, Equations.GraphEquation activationFunc, Operations.ConvOp poolOperation)
		{
			NodeLinkKernels = nodeLinkKernels;
			Bias = bias;
			_activationFunc = activationFunc;
			_poolOperation = poolOperation;
			_poolSampleWH = poolSampleWH;
		}

		public float[,] CalculateData() // TODO Test CalculateData()
		{
			if (_cData != null) return _cData;

			// Calculate all previous node and channel data
			float[][,] allChannelsData = NodeLinkKernels.Select(nlk => nlk.Node.CalculateData()).ToArray(); // float[node][channel][x, y]

			int outW = allChannelsData[0].GetLength(0), outH = allChannelsData.GetLength(1);
			if (!allChannelsData.All(m => m.GetLength(0) == outW && m.GetLength(1) == outH))
				throw new Exception("CalculateData critical error: two input nodes/channels have differing dimensions??? (This shouldn't happen, ever.)");
			float[,] outChannel = new float[outW, outH];

			for (int chanN = 0; chanN < allChannelsData.Length; chanN++) // For channels/nodes
			{
				float[,] pixelData = allChannelsData[chanN];

				// Scan pixelData and get the sum of the components of the dot between the kernel and current focus
				int kernelW = NodeLinkKernels[chanN].Kernel.GetLength(0), kernelH = NodeLinkKernels[chanN].Kernel.GetLength(1);
				int horizSteps = pixelData.GetLength(0) - kernelW + 1;
				int vertSteps = pixelData.GetLength(1) - kernelH + 1;
				float[,] filteredData = new float[horizSteps, vertSteps]; // Width = (w - f)/s + 1

				// Convolute current channel via kernel
				for (int vp = pixelData.GetLength(1) - 1; vp >= kernelH - 1; vp--) // Down pixelData Y
				{
					for (int hp = 0; hp < horizSteps; hp++) // Right pixelData X
					{
						float dotSum = 0;
						// Calculate sum of dot components
						for (int x = 0; x < kernelW; x++) // Right filter X
						{
							for (int y = 0; y < kernelH; y++) // Down filter Y
							{
								dotSum += NodeLinkKernels[chanN].Kernel[x, y] * pixelData[hp + x, vp - y];
							}
						}
						filteredData[hp, vp - kernelH + 1] = _activationFunc(dotSum + Bias);
					}
				}

				// Pool via method the current filteredData matrix
				int pDataLenX = horizSteps / _poolSampleWH, pDataLenY = vertSteps / _poolSampleWH;
				float[,] pooledData = new float[pDataLenX, pDataLenY];
				for (int fDV = _poolSampleWH - 1, pDV = 0; fDV < vertSteps; fDV += _poolSampleWH, pDV++) // Up filteredData Y
				{
					for (int fDH = 0, pDH = 0; fDH < horizSteps; fDH += _poolSampleWH, pDH++) // Right filteredData X
					{
						float[,] sample = new float[_poolSampleWH, _poolSampleWH];
						for (int y = 0; y < _poolSampleWH; y++) // Up local area Y
						{
							for (int x = 0; x < _poolSampleWH; x++) // Right local area X
							{
								sample[x, y] = filteredData[fDH + x, fDV + y];
							}
						}
						pooledData[pDH, pDV] = _poolOperation(sample);
					}
				} // CONTINUE HERE FIX ALL THIS MAN IT NEEDS TO ACTUALLY DAMN WORK LIKE WHAT DA HELL BRUH WHY ONLY NOW REALISE THIS LIKE BRUH AT LEAST IT'S WAY SIMPLER BUT STILL REALLY DAMN ANNOYING

				// Set current output channel
				outChannel[outputIndex] = pooledData;
			}

			return outChannel;
		}
	}

	internal class PolyMatrixInputNodeMono : IMonoConvNeuralNode
	{
		internal float[,] _cachedData = null;
		public float[,] CalculateData()
		{
			if (_cachedData == null)
				throw new NullReferenceException("PolyMatrixInputNodeMono critical error: null cached data on request!");
			return _cachedData;
		}
	}

	internal interface IMonoConvNeuralNode
	{
		/// <returns>float[channel][x, y]</returns>
		public float[,] CalculateData();
	}

	public readonly struct ConvNetworkResult
	{
		public int DecidedType { get; }
		public float CertaintyOfType { get; }
		public Vector2 TopLeftBound { get; }
		public Vector2 BottomRightBound { get; }

		internal ConvNetworkResult(int decidedType, float certaintyOfType, Vector2 topLeftBound, Vector2 bottomRightBound)
		{
			DecidedType = decidedType;
			CertaintyOfType = certaintyOfType;
			TopLeftBound = topLeftBound;
			BottomRightBound = bottomRightBound;
		}
	}
}
