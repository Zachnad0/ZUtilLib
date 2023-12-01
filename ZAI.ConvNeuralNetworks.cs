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
			float[][,] allChannelsData = NodeLinkKernels.Select(nlk => nlk.Node.CalculateData()).ToArray(); // float[channel][x, y]

			int kernelWH = NodeLinkKernels[0].Kernel.GetLength(0); // Can't possibly go wrong, right?
			int chnW = allChannelsData[0].GetLength(0), chnH = allChannelsData.GetLength(1);
			if (!allChannelsData.All(m => m.GetLength(0) == chnW && m.GetLength(1) == chnH))
				throw new Exception("CalculateData critical error: two input nodes/channels have differing dimensions??? (This shouldn't happen, ever.)");
			float[,] convolutedChannel = new float[chnW - kernelWH + 1, chnH - kernelWH + 1];

			// Convolute
			for (int convVert = chnH - kernelWH; convVert >= 0; convVert--) // Top, down (conv grid)
			{
				for (int convHoriz = 0; convHoriz < chnW - kernelWH + 1; convHoriz++) // Left, rightwards (conv grid)
				{
					// Translate to input channel's coordinates
					int inpX = convHoriz, inpY = convVert + kernelWH - 1;
					// Get the sum of the dot-sums from each channel
					float[] dotSums = new float[allChannelsData.Length];
					for (int chanN = 0; chanN < allChannelsData.Length; chanN++)
					{
						dotSums[chanN] = 0;
						for (int x = 0; x < kernelWH; x++) // Left, rightwards (local offset)
						{
							for (int y = kernelWH - 1; y >= 0; y--) // Top, down (local offset)
							{
								dotSums[chanN] += allChannelsData[chanN][inpX + x, inpY - kernelWH + 1 + y] * NodeLinkKernels[chanN].Kernel[x, y];
							}
						}
					}
					convolutedChannel[convHoriz, convVert] = _activationFunc(dotSums.Sum() + Bias);
				}
			}

			// Pool
			float[,] pooledChannel = new float[convolutedChannel.GetLength(0) / _poolSampleWH, convolutedChannel.GetLength(1) / _poolSampleWH];
			for (int poolVert = pooledChannel.GetLength(1) - 1; poolVert >= 0; poolVert--) // Top, down (pool grid)
			{
				for (int poolHoriz = 0; poolHoriz < pooledChannel.GetLength(0); poolHoriz++) // Left, rightwards (pool grid)
				{
					// Translate to convoluted channel's coordinates
					int convX = poolHoriz * _poolSampleWH, convY = poolVert * _poolSampleWH;
					float[,] sample = new float[_poolSampleWH, _poolSampleWH];
					for (int y = _poolSampleWH - 1; y >= 0; y--) // Top, down (local conv offset)
					{
						for (int x = 0; x < _poolSampleWH; x++) // Left, right (local conv offset)
						{
							sample[x, y] = convolutedChannel[convX + x, convY + y]; // Technically not entirely correct, but it's position does not matter for any of the operations.
						}
					}
					pooledChannel[poolHoriz, poolVert] = _poolOperation(sample);
				}
			}

			return pooledChannel;
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
