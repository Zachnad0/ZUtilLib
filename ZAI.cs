using System;
using System.Linq;
using ZUtilLib.ZAI.Saving;

namespace ZUtilLib.ZAI // Random AI stuff here
{
	/// <summary>
	/// Types of neural data node activation functions.
	/// </summary>
	public enum NDNodeActivFunc
	{
		ReLU,
		Sigmoid,
		SoftPlus
	}

	public static class GraphStuff
	{
		/// <summary>
		/// Uses the current graph equation to get y from x.
		/// </summary>
		/// <param name="x">X value.</param>
		/// <returns>Y value.</returns>
		public delegate float GraphEquation(float x);

		/// <summary>
		/// This is used to obtain the activation function based on the type.
		/// </summary>
		/// <param name="type">Neural data node activation function type.</param>
		/// <returns>The delegate of the corresponding function.</returns>
		public static GraphEquation GetEquationFromType(NDNodeActivFunc type)
		{
			return type switch
			{
				NDNodeActivFunc.ReLU => ReLUEquation,
				NDNodeActivFunc.Sigmoid => SigmoidEquation,
				NDNodeActivFunc.SoftPlus => SoftPlusEquation,
				_ => throw new NotImplementedException(),
			};
		}

		public static float ReLUEquation(float x) => MathF.Max(0, x);
		public static float SigmoidEquation(float x) => 1 / (1 + MathF.Exp(-x));
		public static float SoftPlusEquation(float x) => MathF.Log(1 + MathF.Exp(x));
	}

	/// <summary>
	/// A neural network instance
	/// </summary>
	public class NeuralNetwork
	{
		public InputNode[] InputLayer { get; private set; }

		public OutputNode[] OutputLayer { get; private set; }

		/// <summary>
		/// Layer count by height
		/// </summary>
		public NeuralDataNode[,] InternalLayers { get; private set; }

		public NDNodeActivFunc NodeFuncType { get; private set; }

		private bool _is_initialized = false, _output_naming_setup = false;

		/// <summary>
		/// Construct a new neural network instance.
		/// </summary>
		/// <param name="inputHeight">The number of input nodes.</param>
		/// <param name="outputHeight">The number of output nodes.</param>
		/// <param name="internalHeights">The number of hidden nodes per layer.</param>
		/// <param name="internalCount">The number of hidden node layers.</param>
		/// <param name="nodeFuncType">The type of graph to be used for each node's calculation.</param>
		public NeuralNetwork(int inputHeight, int outputHeight, int internalHeights, int internalCount, NDNodeActivFunc nodeFuncType)
		{
			// Set sizes and activation func type
			InputLayer = new InputNode[inputHeight];
			OutputLayer = new OutputNode[outputHeight];
			InternalLayers = new NeuralDataNode[internalCount, internalHeights];
			NodeFuncType = nodeFuncType;
		}

		/// <summary>
		/// This is used to construct a new neural network from a JSON serializable <see cref="PackagedNeuralNetwork"/>. All weights and biases are carried over, and new nodes generated.
		/// </summary>
		/// <param name="packedNet">The neural network weights and biases package to be used in this new NN.</param>
		public NeuralNetwork(PackagedNeuralNetwork packedNet)
		{
			// Is set-up already basically
			_is_initialized = _output_naming_setup = true;

			// Allocation
			InputLayer = new InputNode[packedNet.InputHeight];
			OutputLayer = new OutputNode[packedNet.OutputHeight];
			InternalLayers = new NeuralDataNode[packedNet.InternalCount, packedNet.InternalHeight];
			GraphStuff.GraphEquation activationFunc = GraphStuff.GetEquationFromType(NodeFuncType = packedNet.NodeActivationFunc);

			// Instantiate nodes, assign weights and biases (💀)
			// Inputs
			for (int i = 0; i < packedNet.InputHeight; i++)
			{
				InputLayer[i] = new InputNode(packedNet.InputNodeNames[i]);
			}

			// Internals
			for (int c = 0; c < packedNet.InternalCount; c++)
			{
				for (int i = 0; i < packedNet.InternalHeight; i++)
				{
					(INeuralNode, float)[] links;
					// Allocate then assign links their associated weights and nodes
					if (c == 0)
					{
						links = new (INeuralNode, float)[packedNet.InputHeight];
						for (int p = 0; p < packedNet.InputHeight; p++)
							links[p] = (InputLayer[p], packedNet.InternalNodeWeights[c][i][p]); // 3D moment
					}
					else
					{
						links = new (INeuralNode, float)[packedNet.InternalHeight];
						for (int p = 0; p < packedNet.InternalHeight; p++)
							links[p] = (InternalLayers[c - 1, i], packedNet.InternalNodeWeights[c][i][p]);
					}

					InternalLayers[c, i] = new NeuralDataNode(links, packedNet.InternalNodeBiases[c][i], activationFunc);
				}
			}

			// Outputs
			for (int o = 0; o < packedNet.OutputHeight; o++)
			{
				(INeuralNode, float)[] links = new (INeuralNode, float)[packedNet.InternalHeight];
				// Fill links for this node
				for (int i = 0; i < packedNet.InternalHeight; i++)
				{
					links[i] = (InternalLayers[packedNet.InternalCount - 1, i], packedNet.OutputNodeWeights[o][i]);
				}

				OutputLayer[o] = new OutputNode(links, packedNet.OutputNodeBiases[o], activationFunc, packedNet.OutputNodeNames[o]);
			}
		}

		/// <summary>
		/// Initializes this neural network by with a completely random set of node connection weights.
		/// </summary>
		/// <param name="randomAmplifier">Default 1, changes the max/min random value to be +/- <paramref name="randomAmplifier"/>.</param>
		public void InitializeThis(float randomAmplifier = 1)
		{
			Random rand = new Random();
			GraphStuff.GraphEquation graphEquation = GraphStuff.GetEquationFromType(NodeFuncType);

			float GetRandVal()
			{
				return (randomAmplifier * 2 * (float)rand.NextDouble()) - randomAmplifier;
			}

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
							linkWeights[l] = (InputLayer[l], GetRandVal());

						InternalLayers[c, i] = new NeuralDataNode(linkWeights, GetRandVal(), graphEquation);
					}
					else
					{
						linkWeights = new (INeuralNode, float)[InternalLayers.GetLength(1)];
						for (int l = 0; l < linkWeights.Length; l++)
							linkWeights[l] = (InternalLayers[c - 1, l], GetRandVal());

						InternalLayers[c, i] = new NeuralDataNode(linkWeights, GetRandVal(), graphEquation);
					}
				}
			}

			// Output Layer
			for (int i = 0; i < OutputLayer.Length; i++)
			{
				var linkWeights = new (INeuralNode, float)[InternalLayers.GetLength(1)];
				for (int l = 0; l < linkWeights.Length; l++)
					linkWeights[l] = (InternalLayers[InternalLayers.GetLength(0) - 1, l], GetRandVal());

				OutputLayer[i] = new OutputNode(linkWeights, GetRandVal(), graphEquation);
			}

			_is_initialized = true;
			return;
		}

		/// <summary>
		/// Initializes this neural network based a provided network, mutated randomly through a Gaussian equation.
		/// </summary>
		/// <param name="basedOn">The neural network to be derived from.</param>
		/// <param name="mutationChance">The mutation occurence chance. Should be between 0 and 1.</param>
		/// <param name="learningRate">The severity of any mutations that do occur. Should be between 0 and 1.</param>
		/// <param name="mutateRelative">Optionally, mutate biases by a percentage still influenced by learningRate.</param>
		/// <exception cref="Exception"/>
		public void InitializeThis(NeuralNetwork basedOn, float mutationChance, float learningRate, bool mutateRelative = false)
		{
			Random rand = new Random();
			mutationChance = MathF.Min(MathF.Abs(mutationChance), 1);

			float GuassianChangeEq(float relativeMod)
			{
				float α = MathF.Abs(learningRate); // Bruh
				float χ = (2 * (float)rand.NextDouble()) - 1;
				return Math.Sign(rand.NextDouble() - 0.5d) * (2 * α * MathF.Exp(-2 * MathF.Pow(χ, 2)) - α) * relativeMod;
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
				// Middle layers
				for (int c = 0; c < InternalLayers.GetLength(0); c++)
				{
					for (int i = 0; i < InternalLayers.GetLength(1); i++)
					{
						NeuralDataNode node = InternalLayers[c, i];

						// Links
						for (int l = 0; l < node.LinkNodesWeights.Length; l++)
						{
							var link = node.LinkNodesWeights[l];

							// Modify link weight value IF it should
							if (rand.NextDouble() < mutationChance)
								link.Weight += GuassianChangeEq(mutateRelative ? link.Weight : 1);

							// Re-assign node reference
							if (c == 0) // First layer must replace old references with newer ones in input layer
								link.NeuralNode = InputLayer[l];
							else
								link.NeuralNode = InternalLayers[c - 1, l];

							node.LinkNodesWeights[l] = link;
						}

						// Bias
						if (rand.NextDouble() < mutationChance)
							node.NodeBias += GuassianChangeEq(mutateRelative ? node.NodeBias : 1);
					}
				}

				// Output layer
				foreach (OutputNode node in OutputLayer)
				{
					// Links
					for (int i = 0; i < node.LinkNodesWeights.Length; i++)
					{
						var link = node.LinkNodesWeights[i];

						// Modify link weight value IF it should
						if (rand.NextDouble() < mutationChance)
							link.Weight += GuassianChangeEq(mutateRelative ? link.Weight : 1);

						// Replace link with new instance
						link.NeuralNode = InternalLayers[InternalLayers.GetLength(0) - 1, i];

						node.LinkNodesWeights[i] = link;
					}

					// Bias
					if (rand.NextDouble() < mutationChance)
						node.NodeBias += GuassianChangeEq(mutateRelative ? node.NodeBias : 1);
				}

				_is_initialized = true;
				return;
			}

			throw new Exception("NeuralNetwork InitializeThis Error: Different network scale between this and network to be based on.");
		}

		/// <summary>
		/// <b>ENTIRELY OPTIONAL:</b> Use this after initialization in order to setup the names of the output nodes.
		/// </summary>
		/// <param name="names">The names given to the output nodes (order matters).</param>
		/// <exception cref="Exception"></exception>
		public void SetupOutputs(params string[] names)
		{
			if (_is_initialized && names.Length == OutputLayer.Length)
			{
				for (int i = 0; i < OutputLayer.Length; i++)
					OutputLayer[i].NodeName = new string(names[i]); // The garbage collector will not be pleased.
				return;
			}

			throw new Exception("NeuralNetwork Critical Error: Uninitialized OR too many or too few input node names.");
		}

		/// <summary>
		///<b>USE OVERLOAD FOR UNNAMED OUTPUTS:</b> After initialization and output name setting up, this is used to input the input data and retrieve the calculated outcome.
		/// </summary>
		/// <param name="inputData">The names and values for each input node.</param>
		/// <returns>The names and calculated output values of each output node.</returns>
		/// <exception cref="Exception"></exception>
		public (string NodeName, float Value)[] PerformCalculations(params (string InputNodeName, float Value)[] inputData)
		{
			if (_is_initialized && _output_naming_setup && inputData.Length == InputLayer.Length)
			{
				// Name and assign input nodes
				for (int i = 0; i < inputData.Length; i++)
				{
					// Assign name and value
					var input = inputData[i];
					InputNode node = InputLayer[i];
					node.NodeName = new string(input.InputNodeName);
					node.outVal = input.Value;
				}

				// Cleanse internal and output nodes cached values
				foreach (NeuralDataNode node in InternalLayers)
					((INeuralNode)node).CachedValue = null;
				foreach (OutputNode node in OutputLayer)
					((INeuralNode)node).CachedValue = null;

				// Calculate and return
				return OutputLayer.Select(node => (node.NodeName, ((INeuralNode)node).CalculateValue())).ToArray();
			}

			throw new Exception("NeuralNetwork Critical Error: Invalid inputs and/or values provided OR not initialized OR outputs not named (perhaps use other overload!)");
		}

		/// <summary>
		/// After initialization, this is used to input the input data and retrieve the calculated outcome.
		/// </summary>
		/// <param name="inputData">The names and values for each input node.</param>
		/// <returns>The names and calculated output values of each output node.</returns>
		/// <exception cref="Exception"></exception>
		public float[] PerformCalculations(params float[] inputData)
		{
			if (_is_initialized && inputData.Length == InputLayer.Length)
			{
				// Name and assign input values
				for (int i = 0; i < inputData.Length; i++)
					InputLayer[i].outVal = inputData[i];

				// Cleanse internal and output nodes cached values
				foreach (NeuralDataNode node in InternalLayers)
					((INeuralNode)node).CachedValue = null;
				foreach (OutputNode node in OutputLayer)
					((INeuralNode)node).CachedValue = null;

				// Calculate and return
				return OutputLayer.Select(node => ((INeuralNode)node).CalculateValue()).ToArray();
			}

			throw new Exception("NeuralNetwork Critical Error: Invalid number inputs provided OR not initialized");
		}
	}

	/// <summary>
	/// I mean yea it is called a neuron, but uh skill issue
	/// </summary>
	public class NeuralDataNode : INeuralNode
	{
		public (INeuralNode NeuralNode, float Weight)[] LinkNodesWeights { get; private set; }

		public float NodeBias { get; internal set; }

		float? INeuralNode.CachedValue { get; set; } = null;

		protected GraphStuff.GraphEquation _activationFunc;

		public NeuralDataNode((INeuralNode, float)[] linkWeights, float nodeBias, GraphStuff.GraphEquation activationFunc)
		{
			LinkNodesWeights = linkWeights;
			NodeBias = nodeBias;
			_activationFunc = activationFunc;
		}

		public INeuralNode Clone()
		{
			var newLinkWeights = new (INeuralNode, float)[LinkNodesWeights.Length];
			for (int i = 0; i < LinkNodesWeights.Length; i++)
				newLinkWeights[i] = (LinkNodesWeights[i].NeuralNode, LinkNodesWeights[i].Weight);

			return new NeuralDataNode(newLinkWeights, NodeBias, _activationFunc);
		}

		public override string ToString() => $"NeuralDataNode(B: {NodeBias}, LC: {LinkNodesWeights.Length})";

		float INeuralNode.CalculateValue()
		{
			float output = 0;
			foreach (var link in LinkNodesWeights) // Iterate through, sum of individual output by weight
			{
				INeuralNode linkNode = link.NeuralNode;
				output += link.Weight * (linkNode.CachedValue ?? linkNode.CalculateValue());
			}

			INeuralNode nn = this;
			nn.CachedValue = _activationFunc(output + NodeBias);
			return nn.CachedValue.Value;
		}
	}

	/// <summary>
	/// Derived directly from a normal node, it just has a name property too.
	/// </summary>
	public class OutputNode : NeuralDataNode, INeuralNode
	{
		public string NodeName { get; set; }

		public OutputNode((INeuralNode, float)[] linkWeights, float nodeBias, GraphStuff.GraphEquation activationFunc, string name = "UNNAMED") : base(linkWeights, nodeBias, activationFunc)
		{
			NodeName = name;
		}

		public new OutputNode Clone()
		{
			var newLinkWeights = new (INeuralNode, float)[LinkNodesWeights.Length];
			for (int i = 0; i < LinkNodesWeights.Length; i++)
				newLinkWeights[i] = (LinkNodesWeights[i].NeuralNode, LinkNodesWeights[i].Weight);

			return new OutputNode(newLinkWeights, NodeBias, _activationFunc, new string(NodeName));
		}

		public override string ToString() => $"OutputNode(N: {NodeName}, B: {NodeBias})";
	}

	/// <summary>
	/// This is to be used in the input layer of the NN
	/// </summary>
	public class InputNode : INeuralNode
	{
		internal float? outVal;

		public string NodeName { get; internal set; } // Just for easy debugging

		float? INeuralNode.CachedValue { get => outVal; set => _ = value; }

		public InputNode(string name = "UNNAMED")
		{
			NodeName = name;
		}

		public override string ToString() => $"InputNode(N: {NodeName}, V: {outVal})";

		public INeuralNode Clone()
		{
			return new InputNode(new string(NodeName));
		}

		float INeuralNode.CalculateValue()
		{
			if (outVal.HasValue)
				return outVal.Value;
			throw new Exception("InputNode Critical Error: Input value not provided");
		}
	}

	/// <summary>
	/// All neural nodes implement this.
	/// </summary>
	public interface INeuralNode
	{
		public INeuralNode Clone();
		internal float CalculateValue();
		internal float? CachedValue { get; set; }
	}
}