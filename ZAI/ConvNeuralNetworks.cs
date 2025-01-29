using System;
using System.Collections.Generic;
using System.Linq;
using ZUtilLib.ZAI.FFNeuralNetworks;
using ZUtilLib.ZAI.Saving;

namespace ZUtilLib.ZAI.ConvNeuralNetworks
{
	/// <summary>
	/// An instance of a convolutional neural network.
	/// </summary>
	public class ConvNeuralNetwork
	{
		internal MatrixInputNode[] InputNodes { get; set; }
		/// <summary>[Layer][Node]</summary>
		internal ConvPoolNode[][] ConvPoolNodesLayers { get; set; }
		internal NeuralNetwork FullyConnectedNN { get; set; }

		internal readonly NDNodeActivFunc[] _convFuncLayers;
		internal readonly ConvPoolingOp[] _poolingOpLayers;
		internal readonly Equations.GraphEquation[] _filterActivFuncs;
		internal readonly MatrixOperations.ConvOp[] _poolingMethods;
		internal readonly int[] _kernelWHs, _poolSampleWHs;
		internal readonly NDNodeActivFunc _finalNNActivFunc;
		internal readonly (int W, int H) _inputChannelsSize, _finalNNHiddenLayersCH;
		internal readonly int _inputNodeCount, _outputNodeCount;
		internal bool _initialized = false;

		// TODO (post 3.0.0) add overload that uses an array of layer settings classes for more compact and modular construction
		/// <summary>
		/// Constructs a new instance of <see cref="ConvNeuralNetwork"/> with the specifications provided. This instance must then have one of the initializing methods run in order to be fully initialized.<br/><b>Note:</b> all of the array parameters in this constructor should (and must) be the same length (All are for each conv/pool layer, thus the length of the arrays determines the number of layers).
		/// </summary>
		/// <param name="inputNodeCount">The number of input channels/matrices/nodes this network accepts. <i>(Note: all of the inputs must have the same size/dimensions)</i></param>
		/// <param name="outputNodeCount">The number/size of the output array of floats as a final result from the network's computations.</param>
		/// <param name="inputNodeChannelsSize">The width and height in pixels of the input channels/nodes.</param>
		/// <param name="finalNNHiddenLayersCH">The number of layers, and the height/node count for each layer, of hidden/internal neural nodes of the internal final feed-forward neural network.</param>
		/// <param name="finalNNActivFunc">The internal final feed-forward's node activation function type.</param>
		/// <param name="kernelWHs">The height/width (each one is a square) of the kernels in <b>each conv/pool layer</b>.</param>
		/// <param name="poolSampleWHs">The height/width (each one is a square) of the sample square for <b>each conv/pool layer</b>.</param>
		/// <param name="convAndPoolLayerHeights">The number of conv/pool nodes in <b>each conv/pool layer</b>.</param>
		/// <param name="convActivationFuncs">The activation function applied to the value obtained from each convolution in the conv operation of a conv/pool node, for <b>each conv/pool layer</b>.</param>
		/// <param name="poolingOperations">The pooling operation/method used by every pooling sample in the layer, for <b>each conv/pool layer</b>.</param>
		public ConvNeuralNetwork(int inputNodeCount, int outputNodeCount, (int W, int H) inputNodeChannelsSize, (int Layers, int LayerHeight) finalNNHiddenLayersCH, NDNodeActivFunc finalNNActivFunc, int[] kernelWHs, int[] poolSampleWHs, int[] convAndPoolLayerHeights, NDNodeActivFunc[] convActivationFuncs, ConvPoolingOp[] poolingOperations) // TODO (post 3.0.0) add step settings??? (no pool step though (is auto))
		{
			// Settings verification
			IEnumerable<Array> lengths = new List<Array>() { poolSampleWHs, convAndPoolLayerHeights, convActivationFuncs, poolingOperations }; // It works...
			if (!lengths.All(a => a.Length == kernelWHs.Length))
				throw new Exception("ConvolutionalNeuralNetworkMono critical error: inconsistency with inputted layer setting array lengths.");

			// Fields
			_kernelWHs = kernelWHs;
			_poolSampleWHs = poolSampleWHs;
			_inputNodeCount = inputNodeCount;
			_outputNodeCount = outputNodeCount;
			_finalNNHiddenLayersCH = finalNNHiddenLayersCH;
			_convFuncLayers = convActivationFuncs;
			_poolingOpLayers = poolingOperations;
			_filterActivFuncs = convActivationFuncs.Select(Equations.GetEquationFromType).ToArray();
			_poolingMethods = poolingOperations.Select(MatrixOperations.GetOperationFromType).ToArray();
			_inputChannelsSize = inputNodeChannelsSize;
			_finalNNActivFunc = finalNNActivFunc;

			// Array init
			InputNodes = new MatrixInputNode[inputNodeCount];
			ConvPoolNodesLayers = new ConvPoolNode[convAndPoolLayerHeights.Length][];
			for (int i = 0; i < convAndPoolLayerHeights.Length; i++)
				ConvPoolNodesLayers[i] = new ConvPoolNode[convAndPoolLayerHeights[i]];
		}

		/// <summary>
		/// Un-packages a <see cref="PackagedConvNeuralNetwork"/> into a usable CNN instance.<br/>
		/// Do <b>not</b> run InitializeThis method after.
		/// </summary>
		/// <param name="packedConvNeuralNet">The packaged net to be unpacked</param>
		public ConvNeuralNetwork(PackagedConvNeuralNetwork packedConvNeuralNet)
		{
			// Basic check for argument struct
			if (packedConvNeuralNet.InputNodeCount == 0 || packedConvNeuralNet.KernelsLinksNodesLayers == null)
				throw new ArgumentException("ConvolutionalNeuralNetwork constructor critical error: packedConvNeuralNet argument struct contains invalidities.");

			// Copy readonly data fields
			_initialized = true;
			_convFuncLayers = packedConvNeuralNet.ConvNodeActivationFuncsLayers;
			_poolingOpLayers = packedConvNeuralNet.PoolSampleFuncsLayers;
			_filterActivFuncs = _convFuncLayers.Select(Equations.GetEquationFromType).ToArray();
			_poolingMethods = _poolingOpLayers.Select(MatrixOperations.GetOperationFromType).ToArray();
			_kernelWHs = packedConvNeuralNet.KernelsLinksNodesLayers.Select(la => la[0][0].Length).ToArray();
			_poolSampleWHs = packedConvNeuralNet.PoolSampleDimensionsLayers;
			_inputChannelsSize = packedConvNeuralNet.InputNodeDimensions.ToValueTuple();
			_inputNodeCount = packedConvNeuralNet.InputNodeCount;
			var pFNN = packedConvNeuralNet.PackedFullyConnectedNN;
			_outputNodeCount = pFNN.OutputNodeNames.Length;
			_finalNNActivFunc = pFNN.NodeActivationFunc;
			_finalNNHiddenLayersCH = (pFNN.InternalCount, pFNN.InternalHeight);

			// IMPORTANT: EVERYTHING must be initialized and set IN GIVEN ORDER. No exceptions
			// Setup input nodes
			InputNodes = new MatrixInputNode[_inputNodeCount];
			for (int n = 0; n < _inputNodeCount; n++)
				InputNodes[n] = new MatrixInputNode();

			// Setup conv/pool nodes
			var kLNL = packedConvNeuralNet.KernelsLinksNodesLayers;
			ConvPoolNodesLayers = new ConvPoolNode[kLNL.Length][];
			for (int layerN = 0; layerN < kLNL.Length; layerN++)
			{
				ConvPoolNodesLayers[layerN] = new ConvPoolNode[kLNL[layerN].Length];
				for (int nodeN = 0; nodeN < kLNL[layerN].Length; nodeN++)
				{
					// Generate links
					(IConvNeuralNode Node, float[,] Kernel)[] links = new (IConvNeuralNode Node, float[,] Kernel)[kLNL[layerN][nodeN].Length];
					for (int linkN = 0; linkN < links.Length; linkN++)
					{
						links[linkN] = (layerN == 0
							? (IConvNeuralNode)InputNodes[linkN]
							: ConvPoolNodesLayers[layerN - 1][linkN],
							kLNL[layerN][nodeN][linkN].ToNonJaggedMatrix());
					}

					ConvPoolNodesLayers[layerN][nodeN] = new ConvPoolNode(links, packedConvNeuralNet.BiasNodesLayers[layerN][nodeN], _poolSampleWHs[layerN], _filterActivFuncs[layerN], _poolingMethods[layerN]);
				}
			}

			// Setup & unpack final fully-connected NN
			FullyConnectedNN = new NeuralNetwork(pFNN);
		}

		/// <summary>
		/// Initializes the convolutional neural network by randomly generating all kernels, biases, and weights. Random generation amplitude is determined by <paramref name="initialWeightAmp"/>.<br/>See overload for deriving.
		/// </summary>
		/// <param name="initialWeightAmp">The amplitude of randomly generated values.</param>
		public void InitializeThis(float initialWeightAmp = 1)
		{
			Random random = new Random();
			float GetRandVal() => (float)random.NextDouble() * (initialWeightAmp * 2) - initialWeightAmp;

			// Generate input nodes
			for (int n = 0; n < _inputNodeCount; n++)
				InputNodes[n] = new MatrixInputNode();

			// Generate convolutional & pooling nodes
			for (int layer = 0; layer < ConvPoolNodesLayers.Length; layer++)
			{
				for (int n = 0; n < ConvPoolNodesLayers[layer].Length; n++)
				{
					// Setup links, randomise associated kernels, and normalize them
					(IConvNeuralNode, float[,])[] nodeLinkKernels;
					if (layer == 0)
					{
						nodeLinkKernels = new (IConvNeuralNode, float[,])[InputNodes.Length];
						for (int inpNode = 0; inpNode < nodeLinkKernels.Length; inpNode++)
						{
							nodeLinkKernels[inpNode] = (
								InputNodes[inpNode],
								(float[,])((ZMatrix)random.NextMatrixRect(_kernelWHs[layer], _kernelWHs[layer], true) * initialWeightAmp)
								);
						}
					}
					else
					{
						nodeLinkKernels = new (IConvNeuralNode, float[,])[ConvPoolNodesLayers[layer - 1].Length];
						for (int prevLyrNode = 0; prevLyrNode < nodeLinkKernels.Length; prevLyrNode++)
						{
							nodeLinkKernels[prevLyrNode] = (
								ConvPoolNodesLayers[layer - 1][prevLyrNode],
								(float[,])((ZMatrix)random.NextMatrixRect(_kernelWHs[layer], _kernelWHs[layer], true) * initialWeightAmp)
								);
						}
					}

					ConvPoolNodesLayers[layer][n] = new ConvPoolNode(nodeLinkKernels, GetRandVal(), _poolSampleWHs[layer], _filterActivFuncs[layer], _poolingMethods[layer]);
				}
			}

			// Calculate input length for FullyConnectedNN
			(int Width, int Height) = _inputChannelsSize;
			int lastChannelCount = 0;
			for (int lyrN = 0; lyrN < ConvPoolNodesLayers.Length; lyrN++)
			{
				Width = (Width - _kernelWHs[lyrN] + 1) / _poolSampleWHs[lyrN];
				Height = (Height - _kernelWHs[lyrN] + 1) / _poolSampleWHs[lyrN];
			}
			lastChannelCount = ConvPoolNodesLayers[^1].Length;
			int totalLength = Width * Height * lastChannelCount;
			if (totalLength <= 0)
				throw new Exception("InitializeThis critical error: Invalid dimensions or number of channels or something???");

			// Generate neural network
			FullyConnectedNN = new NeuralNetwork(totalLength, _outputNodeCount, _finalNNHiddenLayersCH.H, _finalNNHiddenLayersCH.W, _finalNNActivFunc);
			FullyConnectedNN.InitializeThis(initialWeightAmp);

			_initialized = true;
		}

		/// <summary>
		/// Initializes the convolutional neural network through copying and mutating from <paramref name="basedOnNet"/>.<br/>See overload for generating random original.
		/// </summary>
		/// <param name="basedOnNet">The CNN for this one to be derived from.</param>
		/// <param name="mutateChance">The chance, in decimal, for a mutation to occur on a given bias, weight, or kernel value.</param>
		/// <param name="learningRate">The amplitude of random mutations that occur randomly.</param>
		public void InitializeThis(ConvNeuralNetwork basedOnNet, float mutateChance, float learningRate)
		{
			// Verify other network specifications, but don't bother too much
			if (basedOnNet == null || !basedOnNet._initialized || basedOnNet.InputNodes.Length != InputNodes.Length || basedOnNet._kernelWHs.Length != _kernelWHs.Length || basedOnNet.ConvPoolNodesLayers.Length != ConvPoolNodesLayers.Length)
				throw new Exception("InitializeThis critical error: basedOnNet differs from the current network's specs in some way.");

			// Init random gen and input nodes
			Random rand = new Random();
			float GetRandAmpVal() => (float)rand.NextDouble() * learningRate * 2 - learningRate;
			for (int i = 0; i < _inputNodeCount; i++)
				InputNodes[i] = new MatrixInputNode();

			// Clone over conv-pool nodes, regenerate their links and kernels with (maybe) slightly modified kernels.
			for (int lN = 0; lN < ConvPoolNodesLayers.Length; lN++)
			{
				for (int nN = 0; nN < ConvPoolNodesLayers[lN].Length; nN++)
				{
					// Copy kernels based on links
					float[][,] kernelClones = basedOnNet.ConvPoolNodesLayers[lN][nN].NodeLinkKernels.Select(t => (float[,])t.Kernel.Clone()).ToArray();
					float bias = basedOnNet.ConvPoolNodesLayers[lN][nN].Bias;
					if (rand.NextDouble() <= mutateChance)
						bias += GetRandAmpVal();

					// Quick check to ensure no differences or anything
					if (kernelClones.Length != (lN == 0 ? _inputNodeCount : ConvPoolNodesLayers[lN - 1].Length))
						throw new Exception("InitializeThis critical error: basedOnNet node has a different number of links on at least one node.");

					// Mutate parts of kernels if necessary
					kernelClones.Foreach((i, arr) => arr.SetEach((x, y, v) => rand.NextDouble() <= mutateChance ? v + GetRandAmpVal() : v));

					// Node link setup
					(IConvNeuralNode, float[,])[] nodeLinkKernels;
					if (lN == 0) // Links to input nodes
					{
						nodeLinkKernels = new (IConvNeuralNode, float[,])[_inputNodeCount];
						for (int iN = 0; iN < _inputNodeCount; iN++)
							nodeLinkKernels[iN] = (InputNodes[iN], kernelClones[iN]);
					}
					else // Links to prev conv-pool layer
					{
						nodeLinkKernels = new (IConvNeuralNode, float[,])[ConvPoolNodesLayers[lN - 1].Length];
						for (int prevN = 0; prevN < nodeLinkKernels.Length; prevN++)
							nodeLinkKernels[prevN] = (ConvPoolNodesLayers[lN - 1][prevN], kernelClones[prevN]);
					}

					ConvPoolNodesLayers[lN][nN] = new ConvPoolNode(nodeLinkKernels, bias, _poolSampleWHs[lN], _filterActivFuncs[lN], _poolingMethods[lN]);
				}
			}

			// Fully-connected NN derivation
			FullyConnectedNN = new NeuralNetwork(basedOnNet.FullyConnectedNN.InputLayer.Length, _outputNodeCount, _finalNNHiddenLayersCH.H, _finalNNHiddenLayersCH.W, _finalNNActivFunc);
			FullyConnectedNN.InitializeThis(basedOnNet.FullyConnectedNN, mutateChance, learningRate);

			_initialized = true;
		}

		/// <summary>
		/// Computes the result this convolutional neural network calculates for the given <paramref name="inputChannels"/>.
		/// </summary>
		/// <param name="inputChannels">The channels to be calculated from. All of these must have the same dimensions and should generally probably be normalized. The length of this array must be the same as the number of input nodes specified in the constructor.</param>
		/// <returns>An array of the computed results, of the length specified in the constructor.</returns>
		public float[] ComputeResultMono(params float[][,] inputChannels)
		{
			if (!_initialized)
				throw new Exception("ComputeResultMono critical error: CNN not initialized.");
			// Check input validity
			if (inputChannels.Length != _inputNodeCount)
				throw new Exception("ComputeResultMono critical error: invalid number of provided input channels.");
			// Check channels size validity
			if (inputChannels.Any(ch => ch.GetLength(0) != _inputChannelsSize.W || ch.GetLength(1) != _inputChannelsSize.H))
				throw new Exception("ComputeResultMono critical error: invalid size of a provided input channel.");

			//// Normalize all channels on all inputs
			//for (int i = 0; i < inputChannels.Length; i++)
			//	inputChannels[i] = inputChannels[i].NormalizeMatrix(false);

			// Clear cached node data, whilst setting inputs
			InputNodes.Foreach((i, n) => n._cachedData = inputChannels[i]);
			ConvPoolNodesLayers.Foreach((x, y, n) => n._cachedData = null);

			// Calculate the final channels, flatten, then pass into final NN for result
			float[] flattenedChannels = new float[FullyConnectedNN.InputLayer.Length];
			int fN = 0;
			for (int nodeN = 0; nodeN < ConvPoolNodesLayers[^1].Length; nodeN++)
			{
				float[,] currChannel = ConvPoolNodesLayers[^1][nodeN].CalculateData();
				currChannel.Foreach((x, y, v) => { flattenedChannels[fN] = v; fN++; });
			}
			float[] finalResult = FullyConnectedNN.PerformCalculations(flattenedChannels);

			return finalResult;
		}
	}

	internal class ConvPoolNode : IConvNeuralNode
	{
		internal float[,] _cachedData = null;
		public (IConvNeuralNode Node, float[,] Kernel)[] NodeLinkKernels { get; private set; }
		public float Bias { get; private set; }

		private readonly Equations.GraphEquation _activationFunc;
		private readonly MatrixOperations.ConvOp _poolOperation;
		private readonly int _poolSampleWH;

		public ConvPoolNode((IConvNeuralNode Node, float[,] Kernel)[] nodeLinkKernels, float bias, int poolSampleWH, Equations.GraphEquation activationFunc, MatrixOperations.ConvOp poolOperation)
		{
			NodeLinkKernels = nodeLinkKernels;
			Bias = bias;
			_activationFunc = activationFunc;
			_poolOperation = poolOperation;
			_poolSampleWH = poolSampleWH;
		}

		public float[,] CalculateData()
		{
			if (_cachedData != null) return _cachedData;

			// Calculate all previous node and channel data
			float[][,] allChannelsData = NodeLinkKernels.Select(nlk => nlk.Node.CalculateData()).ToArray(); // float[channel][x, y]

			int kernelWH = NodeLinkKernels[0].Kernel.GetLength(0); // Can't possibly go wrong, right?
			int chnW = allChannelsData[0].GetLength(0), chnH = allChannelsData[0].GetLength(1);
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
					for (int y = 0; y < _poolSampleWH; y++) // Bottom, up (local conv offset)
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

	internal class MatrixInputNode : IConvNeuralNode
	{
		internal float[,] _cachedData = null;
		public float[,] CalculateData()
		{
			if (_cachedData == null)
				throw new NullReferenceException("PolyMatrixInputNodeMono critical error: null cached data on request!");
			return _cachedData;
		}
	}

	internal interface IConvNeuralNode
	{
		/// <returns>float[channel][x, y]</returns>
		public float[,] CalculateData();
	}
}
