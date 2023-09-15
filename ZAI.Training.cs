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
		/// <returns></returns>
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
			// CONTINUE HERE ===================================================================
		}

		private static bool CheckCompatibleStandardParams(NeuralNetTrainingOptions options, NeuralNetwork[] startNets)
		{
			bool c1 = startNets.All(n =>
			n.InputLayer.Length == options.InputHeight &&
			n.InternalLayers.GetLength(0) == options.InternalCount &&
			n.InternalLayers.GetLength(1) == options.InternalHeight &&
			n.OutputLayer.Length == options.OutputHeight
			);

			return c1;
		}
	}

	public readonly struct NeuralNetTrainingOptions
	{
		public readonly int GenSize, Generations, InputHeight, OutputHeight, InternalHeight, InternalCount, TestsPerNet;
		public readonly float LearningRate, MutateChance, InitialGenAmp, TestRangeMin, TestRangeMax, GenPassCount, MinTargAccuracy;
		public readonly bool MutateRelative, IterateByGenerations;
		// InitialGenAmp, TestRangeMin/Max, and Generations XOR MinTargAccuracy are OPTIONAL

		public NeuralNetTrainingOptions(int genSize, int genPassCount, int testsPerNet, int inputHeight, int outputHeight, int internalHeight, int internalCount, float learningRate, float mutateChance, bool mutateRelative, bool iterateByGenerations, float initialGenAmp = 1, float testRangeMin = 0, float testRangeMax = 1, int generations = 0, float minTargAccuracy = float.PositiveInfinity)
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
