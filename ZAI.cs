using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using System.Linq;
using System.Runtime.CompilerServices;

namespace ZUtilLib.ZAI // Random AI stuff here
{
	/// <summary>
	/// A neural network instance
	/// </summary>
	public class NeuralNetwork
	{
		[JsonPropertyName("input_layer")]
		public InputNode[] InputLayer { get; private set; }

		[JsonPropertyName("output_layer")]
		public OutputNode[] OutputLayer { get; private set; }

		/// <summary>
		/// Layer count by height
		/// </summary>
		[JsonPropertyName("internal_layers")]
		public NeuralDataNode[,] InternalLayers { get; private set; }

		/// <summary>
		/// Construct a new neural network instance.
		/// </summary>
		/// <param name="inputHeight">The number of input nodes.</param>
		/// <param name="outputHeight">The number of output nodes.</param>
		/// <param name="internalHeights">The number of hidden nodes per layer.</param>
		/// <param name="internalCount">The number of hidden node layers.</param>
		public NeuralNetwork(int inputHeight, int outputHeight, int internalHeights, int internalCount)
		{
			// Initialize
			InputLayer = new InputNode[inputHeight];
			OutputLayer = new OutputNode[outputHeight];
			InternalLayers = new NeuralDataNode[internalCount, internalHeights];
		}

		/// <summary>
		/// Initializes this neural network by with a completely random set of node connection weights.
		/// </summary>
		public void InitializeThis()
		{
			Random rand = new Random();

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

		/// <summary>
		/// Initializes this neural network based a provided network, mutated via a Gaussian equation.
		/// </summary>
		/// <param name="basedOn">The neural network to be derived from.</param>
		/// <param name="mutationChance">The mutation occurence chance. Should be between 0 and 1.</param>
		/// <param name="mutationMagnitude">The severity of any mutations that do occur. Should be between 0 and 1.</param>
		/// <exception cref="Exception"/>
		public void InitializeThis(NeuralNetwork basedOn, float mutationChance, float mutationMagnitude)
		{
			Random rand = new Random();
			mutationChance = MathF.Min(MathF.Abs(mutationChance), 1);

			float GuassianEq()
			{
				float α = MathF.Min(MathF.Abs(mutationMagnitude), 1); // Bruh
				float χ = (2 * (float)rand.NextDouble()) - 1;
				return MathF.Sign((float)rand.NextDouble() - 0.5f) * (2 * α * MathF.Exp(-2 * MathF.Pow(χ, 2)) - α);
			}

			if (basedOn.InputLayer.Length == InputLayer.Length && basedOn.InternalLayers.Length == InternalLayers.Length && basedOn.OutputLayer.Length == OutputLayer.Length) // To ensure it's sizing is identical
			{
				// Cloning, VERY IMPORTANT THAT IT IS DONE CORRECTLY
				InputLayer = basedOn.InputLayer.Select(n => (InputNode)n.Clone()).ToArray();
				OutputLayer = basedOn.OutputLayer.Select(n => n.Clone()).ToArray();
				for (int x = 0; x < InternalLayers.GetLength(0); x++) // fr bruh you gotta be kidding me
					for (int y = 0; y < InternalLayers.GetLength(1); y++)
						InternalLayers[x, y] = (NeuralDataNode)basedOn.InternalLayers[x, y].Clone();

				// Iterate through all possible links, roll for chance, then addon itself times GuassianEq()
				void CheckDoModNodeLink(NeuralDataNode node)
				{
					for (int i = 0; i < node.LinkNodesWeights.Length; i++)
					{
						if (rand.NextDouble() < mutationChance)
						{
							var link = node.LinkNodesWeights[i];
							link.Weight += GuassianEq() * link.Weight;
							node.LinkNodesWeights[i] = link;
						}
					}
				}

				// Middle layers
				foreach (NeuralDataNode node in InternalLayers)
					CheckDoModNodeLink(node);

				// Output layer
				foreach (OutputNode node in OutputLayer)
					CheckDoModNodeLink(node);

				return;
			}

			throw new Exception("NeuralNetwork InitializeThis Error: Different network scale between this and network to be based on.");
		}
	}

	/// <summary>
	/// I mean yea it is called a neuron, but uh skill issue
	/// </summary>
	public class NeuralDataNode : INeuralNode
	{
		[JsonPropertyName("input_neurons_link_weights")]
		public (INeuralNode NeuralNode, float Weight)[] LinkNodesWeights { get; private set; }

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

		public INeuralNode Clone()
		{
			var newLinkWeights = new (INeuralNode, float)[LinkNodesWeights.Length];
			for (int i = 0; i < LinkNodesWeights.Length; i++)
				newLinkWeights[i] = (LinkNodesWeights[i].NeuralNode, LinkNodesWeights[i].Weight);

			return new NeuralDataNode(newLinkWeights);
		}
	}

	/// <summary>
	/// Derived directly from a normal node, it just has a name property too.
	/// </summary>
	public class OutputNode : NeuralDataNode, INeuralNode
	{
		[JsonPropertyName("output_node_name")]
		public string NodeName { get; private set; }

		[JsonConstructor]
		public OutputNode((INeuralNode, float)[] linkWeights, string name = "UNNAMED") : base(linkWeights)
		{
			NodeName = name;
		}

		public new OutputNode Clone()
		{
			var newLinkWeights = new (INeuralNode, float)[LinkNodesWeights.Length];
			for (int i = 0; i < LinkNodesWeights.Length; i++)
				newLinkWeights[i] = (LinkNodesWeights[i].NeuralNode, LinkNodesWeights[i].Weight);

			return new OutputNode(newLinkWeights, new string(NodeName));
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

		public INeuralNode Clone()
		{
			return new InputNode(new string(NodeName), outVal);
		}
	}

	/// <summary>
	/// All neural nodes implement this.
	/// </summary>
	public interface INeuralNode
	{
		public float OutputValue { get; }
		public INeuralNode Clone();
	}
}