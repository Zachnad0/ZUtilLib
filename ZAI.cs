using System.Security.Cryptography;
using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using System.Linq;

namespace ZUtilLib.ZAI // Random AI stuff here
{
	/// <summary>
	/// A neural network instance
	/// </summary>
	public class NeuralNetwork
	{
		[JsonPropertyName("input_layer")]
		public NeuralDataUnit[] InputLayer { get; private set; } // These have actually gotta be values

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
		}

		[JsonConstructor]
		public NeuralNetwork(NeuralDataUnit[,] internalLayers)
		{
			// Fill layers from saved data
			InternalLayers = internalLayers;
		}

		public void InitializeThis(NeuralNetwork basedOn = null)
		{
			if (basedOn == null)
			{
				// CONTINUE HERE ===========================================================
				return;
			}

			if (basedOn.InputLayer.Length == InputLayer.Length && basedOn.InternalLayers.Length == InternalLayers.Length && basedOn.OutputLayer.Length == OutputLayer.Length)
			{ // Ensure it's sizing is identical
				return;
			}
		}
	}
}

/// <summary>
/// I mean yea it is called a neuron, but uh skill issue
/// </summary>
public class NeuralDataUnit
{
	[JsonInclude]
	[JsonPropertyName("neuron_weight")]
	public readonly float InternalWeight;

	[JsonPropertyName("input_neurons")]
	public NeuralDataUnit[] InputDataUnits { get; private set; }

	[JsonConstructor]
	public NeuralDataUnit(float neuronValue)
	{
		InternalWeight = neuronValue;
	}
}
