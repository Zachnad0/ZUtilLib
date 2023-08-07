using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using System.Linq;

namespace ZUtilLib.ZAI // Random AI stuff here
{
	/// <summary>
	/// A singular neural network instance
	/// </summary>
	public class NeuralNetwork
	{
		[JsonPropertyName("input_layer")]
		public NeuralDataUnit[] InputLayer { get; private set; }

		[JsonPropertyName("output_layer")]
		public NeuralDataUnit[] OutputLayer { get; private set; }

		/// <summary>
		/// Count by height
		/// </summary>
		[JsonPropertyName("internal_layers")]
		public NeuralDataUnit[,] InternalLayers { get; private set; }

		public NeuralNetwork(int inputHeight, int outputHeight, int internalHeights, int internalCount)
		{
			// Initialize
			InputLayer = new NeuralDataUnit[inputHeight];
			OutputLayer = new NeuralDataUnit[outputHeight];
			InternalLayers = new NeuralDataUnit[internalCount, internalHeights];

			// Generate NeuralDataUnits
			for (int i = 0; i < inputHeight; i++)
			{
				//CONTINUE HERE =====================================================================
			}
		}
	}

	/// <summary>
	/// I mean yea it is called a neuron, but uh skill issue
	/// </summary>
	public class NeuralDataUnit
	{

	}
}
