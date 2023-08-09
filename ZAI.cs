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
			{ // Generate new completely random thing with random biases and etc
				for (int i = 0; i < InputLayer.Length; i++)
				{
					InputLayer[i] = new NeuralDataUnit()
				}

				return;
			}

			if (basedOn.InputLayer.Length == InputLayer.Length && basedOn.InternalLayers.Length == InternalLayers.Length && basedOn.OutputLayer.Length == OutputLayer.Length)
			{ // Ensure it's sizing is identical
				return;
			}
		}
	}

	/// <summary>
	/// I mean yea it is called a neuron, but uh skill issue
	/// </summary>
	public class NeuralDataUnit : INeuralNode
	{
		[JsonPropertyName("input_neurons_link_weights")]
		public (NeuralDataUnit, float)[] LinkUnitsWeights { get; private set; }

		[JsonIgnore]
		public float OutputValue { get => CalculateValue(); }

		[JsonConstructor]
		public NeuralDataUnit((NeuralDataUnit, float)[] linkBias)
		{
			LinkUnitsWeights = linkBias;
		}

		private float CalculateValue()
		{
			foreach (var link in LinkUnitsWeights)
			{
				NeuralDataUnit dataUnit = link.Item1;
				float biasWeight = link.Item2;

				// CONTINUE HERE =============================================
				return default; // Linear regression may be required
			}
		}
	}

	/// <summary>
	/// This is to be used for the input and output layer of the NN
	/// </summary>
	public class SimpleNode : INeuralNode
	{
		[JsonIgnore]
		public float OutputValue { get => outVal.Value; }
		[JsonIgnore]
		private float? outVal;

		public SimpleNode(float? inputValue = null)
		{
			outVal = inputValue;
		}
	}

	public interface INeuralNode
	{
		public float OutputValue { get; }
	}
}