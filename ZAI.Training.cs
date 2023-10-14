using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Threading;

namespace ZUtilLib.ZAI.Training
{
	public delegate float[] TrainingFunctionFormat(params float[] inputs);

	public static class NeuralNetTraining
	{
		private static readonly object _lockOutputs = new object();

		/// <summary>
		/// Generate a generation of networks based on the provided networks and options.
		/// </summary>
		/// <param name="options">Options to define derivation parameters.</param>
		/// <param name="derivedNetworks">Networks for this generation to be derived from.</param>
		/// <returns>Array of size <see cref="NeuralNetTrainingOptions.GenSize"/> as specified in <paramref name="options"/>.</returns>
		public static NeuralNetwork[] GenerateDerivedGeneration(NeuralNetTrainingOptions options, params NeuralNetwork[] derivedNetworks)
		{
			// Check
			if (!CheckCompatibleStandardParams(options, derivedNetworks))
				throw new ArgumentException("Provided networks are invalid.");

			// Generate
			NeuralNetwork[] genNets = new NeuralNetwork[options.GenSize];
			for (int n = 0; n < options.GenSize; n++)
			{
				genNets[n] = new NeuralNetwork(options.InputHeight, options.OutputHeight, options.InternalHeight, options.InternalCount, options.NodeFuncType);
				if (n < derivedNetworks.Length)
					// Exactly clone first networks
					genNets[n].InitializeThis(derivedNetworks[n], 0, 0);
				else
					// Descend from iterated derived network with mutations
					genNets[n].InitializeThis(derivedNetworks[n % derivedNetworks.Length], options.MutateChance, options.LearningRate, options.MutateRelative);
			}

			return genNets;
		}

		/// <summary>
		/// Train this network on a function with <u>strictly</u> the same number of inputs and outputs as the original networks (if provided, otherwise they're randomly generated).
		/// </summary>
		/// <param name="testFunc"></param>
		/// <param name="options"></param>
		/// <param name="startingNetworks"></param>
		/// <returns>The top networks of the last tested generation.</returns>
		public static async Task<NeuralNetwork[]> TrainNetworksOnFunc(TrainingFunctionFormat testFunc, NeuralNetTrainingOptions options, params NeuralNetwork[] startingNetworks)
		{
			// Run checks
			if (!CheckCompatibleStandardParams(options, startingNetworks))
				return null;
			try
			{
				float[] testInputs = new float[options.InputHeight];
				if (testFunc(testInputs).Length != options.OutputHeight)
					throw new Exception();
			}
			catch
			{
				return null;
			}

			// Training
			NeuralNetwork[] topNetsOfGen = startingNetworks.Length > 0 ? startingNetworks : new NeuralNetwork[options.GenPassCount];
			// Skip first random gen if networks are provided
			for (int i = startingNetworks.Length > 0 ? 1 : 0; i < options.Generations; i++)
			{
				// Declare current gen data
				(NeuralNetwork NeuralNet, double NetDiffTotal)[] currentGenData = new (NeuralNetwork NeuralNet, double NetDiffTotal)[options.GenSize];

				// Concurrent network intialization and testing
				currentGenData = await Task.Run(() =>
				{
					(NeuralNetwork NNet, double DiffTotal)[] outputs = new (NeuralNetwork NNet, double DiffTotal)[options.GenSize];

					Parallel.For(0, options.GenSize,
						//new ParallelOptions() { MaxDegreeOfParallelism = 1 },
						(n) =>
					{
						var result = NetTestingOperation(testFunc, options, topNetsOfGen, i, n);
						lock (_lockOutputs)
							outputs[n] = result; // Assign to outputs
					});

					return outputs;
				});

				// Assess and decide best network, the lower the score the better
				List<int> topNetIndices = Enumerable.Range(0, options.GenSize).ToList();
				// Order from lowest to highest score
				topNetIndices.Sort((a, b) =>
				{
					int val;
					try // Yikes I have no idea how but for some reason they can be infinity... Fix this anomaly later???
					{
						val = Math.Sign(currentGenData[a].NetDiffTotal - currentGenData[b].NetDiffTotal);
					}
					catch
					{
						val = currentGenData[a].NetDiffTotal > currentGenData[b].NetDiffTotal ? 1 : -1;
						val = currentGenData[a].NetDiffTotal == currentGenData[b].NetDiffTotal ? 0 : val;
					}
					return val;
				});

				topNetsOfGen = new NeuralNetwork[options.GenPassCount];
				for (int n = 0; n < options.GenPassCount; n++)
					topNetsOfGen[n] = currentGenData[topNetIndices[n]].NeuralNet;
			}

			return topNetsOfGen;
		}

		private static (NeuralNetwork NNet, double DiffTotal) NetTestingOperation(TrainingFunctionFormat testFunc, NeuralNetTrainingOptions options, NeuralNetwork[] topNetsOfGen, int i, int n)
		{
			// Generate network
			NeuralNetwork thisNetwork;
			thisNetwork = new NeuralNetwork(options.InputHeight, options.OutputHeight, options.InternalHeight, options.InternalCount, options.NodeFuncType);

			if (i == 0)
				thisNetwork.InitializeThis(options.InitialGenAmp);
			else if (n < topNetsOfGen.Length) // First whatever count should be exact clones
				thisNetwork.InitializeThis(topNetsOfGen[n], 0, 1);
			else // Alternate based on thing
			{
				int topNetIndex = n % options.GenPassCount;

				thisNetwork.InitializeThis(topNetsOfGen[topNetIndex], options.MutateChance, options.LearningRate, options.MutateRelative);
			}

			// Test and evaluate netowork
			double thisNetDiffTotal = 0;
			for (int t = 0; t < options.TestsPerNet; t++)
			{
				// Iterate and retrieve input values
				float[] testInpVals = new float[options.InputHeight];
				for (int tInp = 0; tInp < options.InputHeight; tInp++)
				{
					if (options.RandomInRange) // Use just a random value if specified
						testInpVals[tInp] = (float)new Random().NextDouble() * (options.TestRangeMax - options.TestRangeMin) + options.TestRangeMin;
					else
						testInpVals[tInp] = (options.TestRangeMax - options.TestRangeMin) / options.TestsPerNet * t + options.TestRangeMin;
				}

				// Iterate through and compare calculated outputs with actual
				float[] testOutVals = thisNetwork.PerformCalculations(testInpVals);
				float[] actualOutVals = testFunc(testInpVals);

				for (int v = 0; v < options.OutputHeight; v++)
					thisNetDiffTotal += Math.Pow(Math.Abs(testOutVals[v] - actualOutVals[v]), 2);
			}
			// Ensure it's validity!
			if (double.IsNaN(thisNetDiffTotal) || double.IsInfinity(thisNetDiffTotal))
				thisNetDiffTotal = float.MaxValue;

			return (thisNetwork, thisNetDiffTotal);
		}

		private static bool CheckCompatibleStandardParams(NeuralNetTrainingOptions options, NeuralNetwork[] startNets)
		{
			bool c1 = startNets.All(n =>
				n.InputLayer.Length == options.InputHeight &&
				n.InternalLayers.GetLength(0) == options.InternalCount &&
				n.InternalLayers.GetLength(1) == options.InternalHeight &&
				n.OutputLayer.Length == options.OutputHeight &&
				n.NodeFuncType == options.NodeFuncType
			);
			bool c2 = startNets?.Length <= options.GenSize;
			return c1 && c2;
		}
	}

	/// <summary>
	/// This class represents all of the options needed for training neural networks, alongside some optional ones.
	/// </summary>
	public class NeuralNetTrainingOptions
	{
		public readonly int GenSize, GenPassCount, Generations, InputHeight, OutputHeight, InternalHeight, InternalCount, TestsPerNet;
		public readonly float LearningRate, MutateChance, InitialGenAmp, TestRangeMin, TestRangeMax, MinTargAccuracy;
		public readonly bool MutateRelative, IterateByGenerations, RandomInRange;
		public readonly NDNodeActivFunc NodeFuncType;
		// InitialGenAmp, TestRangeMin/Max AND RandomInRange, and Generations XOR MinTargAccuracy are OPTIONAL

		/// <summary>
		/// This is for setting all of the options for training neural networks via my methods.
		/// </summary>
		/// <param name="genSize">The number of networks in each generation.</param>
		/// <param name="genPassCount">The number of top networks to be used cloned and varied from, from each generation. Also the number of networks returned.</param>
		/// <param name="testsPerNet">How many tests are done to evaluate each network.</param>
		/// <param name="inputHeight">Number of input nodes.</param>
		/// <param name="outputHeight">Number of output nodes.</param>
		/// <param name="internalHeight">Number of internal nodes per layer.</param>
		/// <param name="internalCount">Number of layers of internal nodes.</param>
		/// <param name="learningRate">The magnitude of network vartaion mutations.</param>
		/// <param name="mutateChance">The chance for each weight and bias to be mutated.</param>
		/// <param name="nodeFuncType">The activation function used by each node within the network.</param>
		/// <param name="mutateRelative">Mutations are done relatively to their current value, instead of straight up.</param>
		/// <param name="iterateByGenerations">If true, then the testing will end after the specified number of networks from parameter <paramref name="generations"/>. If false, then the test will continue until the minimum target accuracy has been achieved as per <paramref name="minTargAccuracy"/>. <u><b>JUST USE TRUE FOR THE TIME BEING STILL WIP FOR THE OTHER ALSO IT IS PROBABLY A TERRIBLE IDEA UNTIL DYNAMIC LEARNING RATE IS ADDED BUT THAT AIN'T NOW SO YEA JUST SET THIS TO TRUE TRUST ME.</b></u></param>
		/// <param name="initialGenAmp">The amplitude of values set for random initial network generation.</param>
		/// <param name="testRangeMin">The minimum value for all inputs used for testing.</param>
		/// <param name="testRangeMax">The maximum value for all outputs used for testing.</param>
		/// <param name="randomInRange">Whether or not for each test to use a randomly generated input value versus an incremental one.</param>
		/// <param name="generations">The number of generations allowed for the training to run.</param>
		/// <param name="minTargAccuracy"><u><b>DON'T USE THIS IN THIS VERSION.</b></u> When the top network of all time reaches this average accuracy value, then the training will end.</param>
		/// <exception cref="ArgumentOutOfRangeException"></exception>
		/// <exception cref="Exception"></exception>
		public NeuralNetTrainingOptions(int genSize, int genPassCount, int testsPerNet, int inputHeight, int outputHeight, int internalHeight, int internalCount, float learningRate, float mutateChance, NDNodeActivFunc nodeFuncType, bool mutateRelative, bool iterateByGenerations, float initialGenAmp = 1, float testRangeMin = 0, float testRangeMax = 1, bool randomInRange = true, int generations = 0, float minTargAccuracy = float.PositiveInfinity)
		{
			// Essentials
			GenSize = genSize;
			GenPassCount = genPassCount > 0 ? genPassCount : throw new ArgumentOutOfRangeException();
			TestsPerNet = testsPerNet;
			InputHeight = inputHeight;
			OutputHeight = outputHeight;
			InternalHeight = internalHeight;
			InternalCount = internalCount;
			LearningRate = learningRate;
			MutateChance = mutateChance;
			NodeFuncType = nodeFuncType;
			MutateRelative = mutateRelative;
			IterateByGenerations = iterateByGenerations;

			// Optional stuff
			InitialGenAmp = initialGenAmp > 0 ? initialGenAmp : throw new ArgumentOutOfRangeException();
			RandomInRange = randomInRange;
			TestRangeMin = testRangeMin;
			TestRangeMax = testRangeMin < testRangeMax ? testRangeMax : throw new ArgumentOutOfRangeException();
			if (iterateByGenerations)
			{
				Generations = generations > 0 ? generations : throw new Exception("IterateByGenerations is true but generations is unset/invalid");
			}
			else
			{
				MinTargAccuracy = minTargAccuracy > 0 && minTargAccuracy != float.PositiveInfinity ? minTargAccuracy : throw new Exception("IterateByGenerations is false but MinTargetAccuracy is unset/invalid");
			}
		}
	}
}
