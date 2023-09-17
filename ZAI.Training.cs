using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using ZUtilLib.ZAI.Saving;

namespace ZUtilLib.ZAI.Training
{
	public delegate float[] TrainingFunctionFormat(params float[] inputs);

	public static class NeuralNetTraining
	{
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

			return await Task.Run(() =>
			{
				// Training
				NeuralNetwork[] topNetsOfGen = startingNetworks.Length > 0 ? startingNetworks : new NeuralNetwork[options.GenPassCount];
				// Skip first random gen if networks are provided
				for (int i = startingNetworks.Length > 0 ? 1 : 0; i < options.Generations; i++)
				{
					// Cleanse memory refs
					double[] currentGenDiffTotals = new double[options.GenSize];
					NeuralNetwork[] currentGenNNs = new NeuralNetwork[options.GenSize];

					// Fill generation with networks
					int indexCycler = 0;
					for (int n = 0; n < options.GenSize; n++)
					{
						currentGenNNs[n] = new NeuralNetwork(options.InputHeight, options.OutputHeight, options.InternalHeight, options.InternalCount, options.NodeFuncType);

						if (i == 0)
							currentGenNNs[n].InitializeThis(options.InitialGenAmp);
						else if (n < topNetsOfGen.Length) // First whatever count should be exact clones
							currentGenNNs[n].InitializeThis(topNetsOfGen[n], 0, 1);
						else // Alternate based on indexCycler
						{
							currentGenNNs[n].InitializeThis(topNetsOfGen[indexCycler], options.MutateChance, options.LearningRate, options.MutateRelative);

							indexCycler++;
							if (indexCycler >= options.GenPassCount)
								indexCycler = 0;
						}
					}

					// Individually assess and tally score based off secret equation
					for (int n = 0; n < options.GenSize; n++)
					{
						currentGenDiffTotals[n] = 0;

						for (int t = 0; t < options.TestsPerNet; t++)
						{
							// Iterate and retrieve input values
							float[] testXVals = new float[options.InputHeight];
							for (int tInp = 0; tInp < options.InputHeight; tInp++)
							{
								testXVals[tInp] = (options.TestRangeMax - options.TestRangeMin) / options.TestsPerNet * t + options.TestRangeMin;
							}

							// Iterate and addon calculated outputs
							for (int tOut = 0; tOut < options.OutputHeight; tOut++)
								currentGenDiffTotals[n] += Math.Pow(Math.Abs(currentGenNNs[n].PerformCalculations(testXVals)[tOut] - testFunc(testXVals)[tOut]), 2);
						}
					};

					// Assess and decide best network, the lower the score the better
					List<int> topNetIndices = Enumerable.Range(0, options.GenSize).ToList();
					// Order from lowest to highest score
					topNetIndices.Sort((a, b) =>
					{
						int val;
						try // Yikes I have no idea how but for some reason they can be infinity... Fix this anomaly later???
						{
							val = Math.Sign(currentGenDiffTotals[a] - currentGenDiffTotals[b]);
						}
						catch
						{
							val = currentGenDiffTotals[a] > currentGenDiffTotals[b] ? 1 : -1;
							val = currentGenDiffTotals[a] == currentGenDiffTotals[b] ? 0 : val;
						}
						return val;
					});
					//topNetIndices.RemoveRange(options.GenPassCount, topNetIndices.Count - options.GenPassCount);

					topNetsOfGen = new NeuralNetwork[options.GenPassCount];
					for (int n = 0; n < options.GenPassCount; n++)
						topNetsOfGen[n] = currentGenNNs[topNetIndices[n]];
				}

				return topNetsOfGen;
			});
		}

		private static bool CheckCompatibleStandardParams(NeuralNetTrainingOptions options, NeuralNetwork[] startNets)
		{
			bool c1 = startNets?.All(n =>
				n.InputLayer.Length == options.InputHeight &&
				n.InternalLayers.GetLength(0) == options.InternalCount &&
				n.InternalLayers.GetLength(1) == options.InternalHeight &&
				n.OutputLayer.Length == options.OutputHeight &&
				n.NodeFuncType == options.NodeFuncType
			) ?? false;
			bool c2 = startNets?.Length <= options.GenSize && startNets?.Length > 0;
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
		public readonly bool MutateRelative, IterateByGenerations;
		public readonly NDNodeActivFunc NodeFuncType;
		// InitialGenAmp, TestRangeMin/Max, and Generations XOR MinTargAccuracy are OPTIONAL

		public NeuralNetTrainingOptions(int genSize, int genPassCount, int testsPerNet, int inputHeight, int outputHeight, int internalHeight, int internalCount, float learningRate, float mutateChance, NDNodeActivFunc nodeFuncType, bool mutateRelative, bool iterateByGenerations, float initialGenAmp = 1, float testRangeMin = 0, float testRangeMax = 1, int generations = 0, float minTargAccuracy = float.PositiveInfinity)
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
			TestRangeMin = testRangeMin;
			TestRangeMax = testRangeMin < testRangeMax ? testRangeMax : throw new ArgumentOutOfRangeException();
			Generations = iterateByGenerations && generations > 0 ? generations : throw new Exception("IterateByGenerations is true but generations is unset/invalid");
			MinTargAccuracy = !iterateByGenerations && minTargAccuracy > 0 && minTargAccuracy != float.PositiveInfinity ? minTargAccuracy : throw new Exception("IterateByGenerations is false but MinTargetAccuracy is unset/invalid");
		}
	}
}
