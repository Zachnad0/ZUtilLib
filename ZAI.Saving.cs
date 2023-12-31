﻿using System.Linq;
using System.Text.Json.Serialization;
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
}
