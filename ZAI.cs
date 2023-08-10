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
		public InputNode[] InputLayer { get; private set; } // These have actually gotta be values

		[JsonPropertyName("output_layer")]
		public OutputNode[] OutputLayer { get; private set; }

		/// <summary>
		/// Layer count by height
		/// </summary>
		[JsonPropertyName("internal_layers")]
		public NeuralDataNode[,] InternalLayers { get; private set; }

		public NeuralNetwork(int inputHeight, int outputHeight, int internalHeights, int internalCount)
		{
			// Initialize
			InputLayer = new InputNode[inputHeight];
			OutputLayer = new OutputNode[outputHeight];
			InternalLayers = new NeuralDataNode[internalCount, internalHeights];
		}

		public NeuralNetwork(NeuralNetwork copyFrom)
		{
			InputLayer = copyFrom.InputLayer;
			OutputLayer = copyFrom.OutputLayer;
			InternalLayers = copyFrom.InternalLayers;
		}

		public void InitializeThis(NeuralNetwork basedOn = null)
		{
			Random rand = new Random();

			if (basedOn == null)
			{ // Generate new completely random thing with random biases and etc
			  // Input layer
				for (int i = 0; i < InputLayer.Length; i++)
				{
					InputLayer[i] = new InputNode();
				}

				// Middle Layers
				for (int c = 0; c < InternalLayers.GetLength(0); c++) // c++ reference???
				{
					for (int i = 0; i < InternalLayers.GetLength(1); i++)
					{
						// Generate links with random weights
						(INeuralNode, float)[] linkWeights;
						if (c == 0)
						{
							linkWeights = new (INeuralNode, float)[InputLayer.Length];
							for (int l = 0; l < InputLayer.Length; l++)
								linkWeights[l] = (InputLayer[l], (float)rand.NextDouble());
							InternalLayers[c, i] = new NeuralDataNode(linkWeights);
						}
						else
						{
							linkWeights = new (INeuralNode, float)[InternalLayers.GetLength(1)];
							for (int l = 0; l < linkWeights.Length; l++)
								linkWeights[l] = (InternalLayers[c - 1, l], (float)rand.NextDouble());
							InternalLayers[c, i] = new NeuralDataNode(linkWeights);
						}
					}
				}

				// Output Layer
				for (int i = 0; i < OutputLayer.Length; i++)
				{
					var linkWeights = new (INeuralNode, float)[InternalLayers.GetLength(1)];
					for (int l = 0; l < linkWeights.Length; l++)
						linkWeights[l] = (InternalLayers[InternalLayers.GetLength(0) - 1, l], (float)rand.NextDouble());
					OutputLayer[i] = new OutputNode(linkWeights);
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
	public class NeuralDataNode : INeuralNode
	{
		[JsonPropertyName("input_neurons_link_weights")]
		public (INeuralNode, float)[] LinkNodesWeights { get; private set; }

		[JsonIgnore]
		public float OutputValue
		{
			get // Calculate output value
			{
				float output = 0;
				foreach (var link in LinkNodesWeights) // Iterate through, sum of individual output by weight
				{
					INeuralNode dataUnit = link.Item1;
					float biasWeight = link.Item2;
					output += biasWeight * dataUnit.OutputValue;
				}
				return output;
			}
		}

		[JsonConstructor]
		public NeuralDataNode((INeuralNode, float)[] linkWeights)
		{
			LinkNodesWeights = linkWeights;
		}
	}

	/// <summary>
	/// Derived directly from a normal node, it just has a name property too.
	/// </summary>
	public class OutputNode : NeuralDataNode
	{
		[JsonPropertyName("output_node_name")]
		public string NodeName { get; private set; }

		[JsonConstructor]
		public OutputNode((INeuralNode, float)[] linkWeights, string name = "UNNAMED") : base(linkWeights)
		{
			NodeName = name;
		}
	}

	/// <summary>
	/// This is to be used in the input layer of the NN
	/// </summary>
	public class InputNode : INeuralNode
	{
		[JsonIgnore]
		public float OutputValue { get => outVal.Value; }
		[JsonIgnore]
		private float? outVal;
		[JsonPropertyName("input_node_name")]
		public string NodeName { get; private set; } // Just for easy debugging

		[JsonConstructor]
		public InputNode(string name = "UNNAMED", float? inputValue = null)
		{
			NodeName = name;
			outVal = inputValue;
		}

		public override string ToString() => $"InputNode({NodeName}, {outVal})";
	}

	public interface INeuralNode
	{
		public float OutputValue { get; }
	}
}