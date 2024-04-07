using System.Linq;
using System.Text.Json.Serialization;
using ZUtilLib.ZAI.ConvNeuralNetworks;
using ZUtilLib.ZAI.FFNeuralNetworks;

namespace ZUtilLib.ZAI.Saving
{
	public readonly struct PackagedNeuralNetwork
	{
		public NDNodeActivFunc NodeActivationFunc { get; }

		// Node naming
		public string[] InputNodeNames { get; }
		public string[] OutputNodeNames { get; }

		// Weights and biases
		public float[][][] InternalNodeWeights { get; } // Non-jagged multidimensional arrays are not supported by the Json Serialization
		public float[][] InternalNodeBiases { get; }
		public float[][] OutputNodeWeights { get; }
		public float[] OutputNodeBiases { get; }

		// Network Sizing
		public int InputHeight { get; }
		public int InternalHeight { get; }
		public int InternalCount { get; }
		public int OutputHeight { get; }

		/// <summary>
		/// Use this to construct a packaged structure ready for JSON serialization or whatever.
		/// </summary>
		/// <param name="neuralNetwork">The neural netowork to have it's data cloned.</param>
		public PackagedNeuralNetwork(NeuralNetwork neuralNetwork)
		{
			NodeActivationFunc = neuralNetwork.NodeFuncType;
			// Naming
			InputNodeNames = neuralNetwork.InputLayer.Select(n => new string(n.NodeName)).ToArray();
			OutputNodeNames = neuralNetwork.OutputLayer.Select(n => new string(n.NodeName)).ToArray();

			// Network Sizing
			InputHeight = neuralNetwork.InputLayer.Length;
			InternalCount = neuralNetwork.InternalLayers.GetLength(0);
			InternalHeight = neuralNetwork.InternalLayers.GetLength(1);
			OutputHeight = neuralNetwork.OutputLayer.Length;

			// Weights and biases
			// Allocate
			InternalNodeBiases = new float[InternalCount][];
			InternalNodeWeights = new float[InternalCount][][];
			OutputNodeBiases = new float[OutputHeight];
			OutputNodeWeights = new float[OutputHeight][];

			// Assign
			// Internal nodes
			for (int c = 0; c < InternalCount; c++)
			{
				InternalNodeBiases[c] = new float[InternalHeight];
				InternalNodeWeights[c] = new float[InternalHeight][];

				for (int h = 0; h < InternalHeight; h++)
				{
					InternalNodeBiases[c][h] = neuralNetwork.InternalLayers[c, h].NodeBias;
					var links = neuralNetwork.InternalLayers[c, h].LinkNodesWeights;
					InternalNodeWeights[c][h] = links.Select(n => n.Weight).ToArray();
				}
			}

			// Output nodes
			for (int o = 0; o < OutputHeight; o++)
			{
				OutputNodeBiases[o] = neuralNetwork.OutputLayer[o].NodeBias;
				OutputNodeWeights[o] = neuralNetwork.OutputLayer[o].LinkNodesWeights.Select(n => n.Weight).ToArray();
			}
		}

		/// <summary>
		/// If you are able to read this, you should use the <u><b>other overload constructor</b></u>.
		/// </summary>
		[JsonConstructor]
		public PackagedNeuralNetwork(NDNodeActivFunc nodeActivationFunc, string[] inputNodeNames, string[] outputNodeNames, float[][][] internalNodeWeights, float[][] internalNodeBiases, float[][] outputNodeWeights, float[] outputNodeBiases, int inputHeight, int internalHeight, int internalCount, int outputHeight)
		{
			NodeActivationFunc = nodeActivationFunc;
			InputNodeNames = inputNodeNames;
			OutputNodeNames = outputNodeNames;
			InternalNodeWeights = internalNodeWeights;
			InternalNodeBiases = internalNodeBiases;
			OutputNodeWeights = outputNodeWeights;
			OutputNodeBiases = outputNodeBiases;
			InputHeight = inputHeight;
			InternalHeight = internalHeight;
			InternalCount = internalCount;
			OutputHeight = outputHeight;
		}
	}

	public readonly struct PackagedConvNeuralNetwork
	{
		public int InputNodeCount { get; }
		public (int W, int H) InputNodeDimensions { get; }
		/// <summary>
		/// float[layerN][nodeN][linkN][kernelXN][kernelYN]
		/// Additionally stores information on the size of these parameters
		/// </summary>
		public float[][][][][] KernelsLinksNodesLayers { get; }
		/// <summary>
		/// float[layerN][nodeN]
		/// </summary>
		public float[][] BiasNodesLayers { get; }
		public int[] PoolSampleDimensionsLayers { get; }
		public NDNodeActivFunc[] ConvNodeActivationFuncsLayers { get; }
		public ConvPoolingOp[] PoolSampleFuncsLayers { get; }
		public PackagedNeuralNetwork PackedFullyConnectedNN { get; }

		public PackagedConvNeuralNetwork(ConvolutionalNeuralNetwork convNeuralNet)
		{
			// Input node specs (no uniquely generated data)
			InputNodeCount = convNeuralNet._inputNodeCount;
			InputNodeDimensions = convNeuralNet._inputChannelsSize;

			// Conv/pool node specs (plus unique kernels, pool sample dim., and biases)
			PoolSampleDimensionsLayers = (int[])convNeuralNet._poolSampleWHs.Clone();

			int convPoolNodeLayerCount = convNeuralNet.ConvAndPoolNodes.Length;
			KernelsLinksNodesLayers = new float[convPoolNodeLayerCount][][][][];
			BiasNodesLayers = new float[convPoolNodeLayerCount][];

			// Iterate through each layer of connv/pool nodes
			for (int layerN = 0; layerN < convPoolNodeLayerCount; layerN++)
			{
				int nodesInLayerCount = convNeuralNet.ConvAndPoolNodes[layerN].Length;
				KernelsLinksNodesLayers[layerN] = new float[nodesInLayerCount][][][];
				BiasNodesLayers[layerN] = new float[nodesInLayerCount];

				// Retrieve node specifics for each node
				for (int nodeN = 0; nodeN < nodesInLayerCount; nodeN++)
				{
					FilterPoolNodeMono currNode = convNeuralNet.ConvAndPoolNodes[layerN][nodeN];
					int nodeLinksCount = currNode.NodeLinkKernels.Length;
					KernelsLinksNodesLayers[layerN][nodeN] = new float[nodeLinksCount][][];

					// Copy kernel for each link
					for (int linkN = 0; linkN < nodeLinksCount; linkN++)
					{
						KernelsLinksNodesLayers[layerN][nodeN][linkN] = currNode.NodeLinkKernels[linkN].Kernel.ToJaggedMatrix();
					}

					// Retrieve bias
					BiasNodesLayers[layerN][nodeN] = currNode.Bias;
				}
			}

			// This theoretically works, but I ain't risking it:
			//KernelsLinksNodesLayers = convNeuralNet.ConvAndPoolNodes.Select(la => la.Select(no => no.NodeLinkKernels.Select(li => li.Kernel.ToJaggedMatrix()).ToArray()).ToArray()).ToArray();

			// Activation and pooling functions
			ConvNodeActivationFuncsLayers = (NDNodeActivFunc[])convNeuralNet._convFuncLayers.Clone();
			PoolSampleFuncsLayers = (ConvPoolingOp[])convNeuralNet._poolingOpLayers.Clone();

			// "Final neural network" which is just packaged in itself
			PackedFullyConnectedNN = new PackagedNeuralNetwork(convNeuralNet.FullyConnectedNN);
		}

		[JsonConstructor]
		public PackagedConvNeuralNetwork(int inputNodeCount, (int W, int H) inputNodeDimensions, float[][][][][] kernelsLinksNodesLayers, float[][] biasNodesLayers, int[] poolSampleDimensionsLayers, NDNodeActivFunc[] convNodeActivationFuncsLayers, ConvPoolingOp[] poolSampleFuncsLayers, PackagedNeuralNetwork packedFullyConnectedNN)
		{
			InputNodeCount = inputNodeCount;
			InputNodeDimensions = inputNodeDimensions;
			KernelsLinksNodesLayers = kernelsLinksNodesLayers;
			BiasNodesLayers = biasNodesLayers;
			PoolSampleDimensionsLayers = poolSampleDimensionsLayers;
			ConvNodeActivationFuncsLayers = convNodeActivationFuncsLayers;
			PoolSampleFuncsLayers = poolSampleFuncsLayers;
			PackedFullyConnectedNN = packedFullyConnectedNN;
		}
	}
}
