using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace ZUtilLib.ZAI.ConvNeuralNetworks
{
	public class ConvolutionalNeuralNetworkMono
	{
		private PolyMatrixInputNodeMono[] InputNodes { get; set; }
		private FilterPoolNodeMono[][] FilterAndPoolNodes { get; set; }
		private Equations.GraphEquation[] _filterActivFuncs;
		private Operations.ConvOp[] _poolingMethods;
		private int[] _kernelWHs, _poolSampleWHs;
		private int _inputNodeCount;
		private bool _initialized = false;

		public ConvolutionalNeuralNetworkMono(int inputChannelCount, int[] kernelWHs, int[] poolSampleWHs, int[] convAndPoolLayerHeights, NDNodeActivFunc[] convActivationFuncs, ConvPoolingOp[] poolingOperations) // TODO add step settings???
		{
			// Settings verification
			var lengths = new List<Array>() { poolSampleWHs, convAndPoolLayerHeights, convActivationFuncs, poolingOperations }; // TODO test to even make sure this works
			if (!lengths.All(a => a.Length == kernelWHs.Length))
				throw new Exception("ConvolutionalNeuralNetworkMono critical error: inconsistency with inputted layer setting array lengths.");

			// Setup
			_kernelWHs = kernelWHs;
			_inputNodeCount = inputChannelCount;
			_filterActivFuncs = convActivationFuncs.Select(t => Equations.GetEquationFromType(t)).ToArray();
			_poolingMethods = poolingOperations.Select(t => Operations.GetOperationFromType(t)).ToArray();

			InputNodes = new PolyMatrixInputNodeMono[inputChannelCount];
			FilterAndPoolNodes = new FilterPoolNodeMono[convAndPoolLayerHeights.Length][];
			for (int i = 0; i < convAndPoolLayerHeights.Length; i++)
				FilterAndPoolNodes[i] = new FilterPoolNodeMono[convAndPoolLayerHeights[i]];


		}

		public void InitializeThis(float mutateChance, float learningRate)
		{
			// CONTINUE HERE with the randomised generation and derived generation below
			_initialized = true;
		}

		public void InitializeThis(ConvolutionalNeuralNetworkMono basedOnNet, float mutateChance, float learningRate)
		{
		}

		public ConvNetworkResult ComputeResultMono(params float[][,] monoPixelGrids)
		{
			// Check input validity
			if (monoPixelGrids.Length != _inputNodeCount)
				throw new Exception("ComputeResultMono critical error: invalid number of provided input channels");

			for (int i = 0; i < monoPixelGrids.Length; i++)
				monoPixelGrids[i] = monoPixelGrids[i].NormalizeMatrix(false);

			return default;
		}
	}

	internal class FilterPoolNodeMono : IMonoConvNeuralNode
	{
		public float[][,] CachedData { get => _cData ?? CalculateData(); set => _cData = value; }
		private float[][,] _cData = null;
		public (IMonoConvNeuralNode Node, float[,] Kernel)[] NodeLinkKernels { get; private set; }
		public float Bias { get; private set; }

		private Equations.GraphEquation _activationFunc;

		public FilterPoolNodeMono((IMonoConvNeuralNode Node, float[,] Kernel)[] nodeLinkKernels, float bias, Equations.GraphEquation activationFunc)
		{
			NodeLinkKernels = nodeLinkKernels;
			Bias = bias;
			_activationFunc = activationFunc;
		}

		public float[][,] CalculateData()
		{
			if (_cData != null) return _cData;

			// Calculate all previous node and channel data
			float[][][,] allChannelsData = NodeLinkKernels.Select(nlk => nlk.Node.CachedData).ToArray(); // float[node][channel][x, y]

			// Get channel count and output channel array size
			int channelCount = 0;
			for (int i = 0; i < allChannelsData.Length; i++)
				channelCount += allChannelsData[i].Length;
			float[][,] outChannels = new float[channelCount][,];

			int outputIndex = 0;
			for (int nodeN = 0; nodeN < allChannelsData.Length; nodeN++) // For nodes
			{
				for (int chanN = 0; chanN < allChannelsData[nodeN].Length; chanN++) // For channels per node
				{
					float[,] pixelData = allChannelsData[nodeN][chanN];

					// Scan pixelData and get the sum of the components of the dot between the kernel and current focus
					int kernelW = NodeLinkKernels[nodeN].Kernel.GetLength(0), kernelH = NodeLinkKernels[nodeN].Kernel.GetLength(1);
					int horizSteps = pixelData.GetLength(0) - kernelW + 1;
					int vertSteps = pixelData.GetLength(1) - kernelH + 1;
					outChannels[outputIndex] = new float[horizSteps, vertSteps]; // Width = (w - f)/s + 1

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
									dotSum += NodeLinkKernels[nodeN].Kernel[x, y] * pixelData[hp + x, vp - y];
								}
							}
							outChannels[outputIndex][hp, vp - kernelH + 1] = _activationFunc(dotSum + Bias);
						}
					}

					// Pool via max of sample size
					// CONTINUE HERE with pool sampling
					outChannels[outputIndex] = outChannels[outputIndex].NormalizeMatrix(false); // Maybe?

					outputIndex++;
				}
			}

			return outChannels;
		}
	}

	internal class PolyMatrixInputNodeMono : IMonoConvNeuralNode
	{
		public float[][,] CachedData { get; set; }

		public float[][,] CalculateData()
		{
			if (CachedData == null)
				throw new NullReferenceException("PolyMatrixInputNodeMono critical error: null cached data on request!");
			return CachedData;
		}
	}

	internal interface IMonoConvNeuralNode
	{
		public float[][,] CachedData { get; set; }
		/// <returns>float[channel][x, y]</returns>
		public float[][,] CalculateData();
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
