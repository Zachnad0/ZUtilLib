using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Threading.Tasks;
using ZUtilLib;
using ZUtilLib.ZAI;
using ZUtilLib.ZAI.ConvNeuralNetworks;
using ZUtilLib.ZAI.Saving;
using ZUtilLib.ZAI.Training;

namespace LIBRARYTESTING
{
	public class Program
	{
		//public static async Task Main(string[] args)
		//{
		//	do
		//	{
		//		Console.WriteLine("Start");

		//		//NeuralNetwork testNet = new NeuralNetwork(3, 3, 5, 2, NDNodeActivFunc.ReLU);
		//		//testNet.InitializeThis();
		//		//testNet.SetupOutputs("alpha", "beta", "gamma");
		//		//(string NodeName, float Value)[] result = testNet.PerformCalculations(("in1", 0.2f), ("in2", 0.3f), ("in3", 0.4f));
		//		//NeuralNetwork secondTestNet = new NeuralNetwork(3, 3, 5, 2, NDNodeActivFunc.ReLU);
		//		//secondTestNet.InitializeThis(testNet, 1, 0.1f, true);
		//		//secondTestNet.SetupOutputs("delta", "epsilon", "zeta");
		//		//var result2 = secondTestNet.PerformCalculations(("in1", 0.2f), ("in2", 0.3f), ("in3", 0.4f));

		//		//PackagedNeuralNetwork pNN1 = new PackagedNeuralNetwork(testNet);
		//		//PackagedNeuralNetwork pNN2 = new PackagedNeuralNetwork(secondTestNet);

		//		//NeuralNetwork cn1, cn2;
		//		//cn1 = new NeuralNetwork(pNN1);
		//		//cn2 = new NeuralNetwork(pNN2);

		//		DateTime start = DateTime.Now;
		//		NeuralNetTrainingOptions nNTO = new NeuralNetTrainingOptions(100, 5, 70, 1, 1, 4, 4, 1, 0.6f, NDNodeActivFunc.ReLU, false, true, 2.5f, -10, 10, randomInRange: true, generations: 1000);
		//		NeuralNetwork[] topNets = await NeuralNetTraining.TrainNetworksOnFunc(MysteryFunc, nNTO);
		//		TimeSpan timeForOperation = DateTime.Now - start;
		//		Console.WriteLine($"Time for operation: {(decimal)timeForOperation.TotalSeconds} seconds");

		//		// Check effectiveness of top networks
		//		for (int n = 0; n < topNets.Length; n++)
		//		{
		//			float diff = 0, highestDiff = float.NegativeInfinity, lowestDiff = float.PositiveInfinity;
		//			for (int i = 0; i < 100; i++)
		//			{
		//				float x = (10 - -10) / 100f * i + -10f;
		//				float y1 = topNets[n].PerformCalculations(x)[0];
		//				float y2 = MysteryFunc(x)[0];
		//				float d = MathF.Abs(y1 - y2);
		//				diff += d;
		//				if (d > highestDiff)
		//					highestDiff = d;
		//				if (d < lowestDiff)
		//					lowestDiff = d;
		//			}
		//			diff /= 100;
		//			Console.WriteLine($"TopNet#{n}:\tAvgDiff: {diff:F4}\t\tHighestDiff: {highestDiff:F4}\tLowestDiff: {lowestDiff:F4}");
		//		}

		//		Console.WriteLine("Finished\n[R]estart?");
		//	} while (Console.ReadKey(true).KeyChar.ToString().ToLower() == "r");
		//}

		/// <summary>
		/// f(x) = 0.002x^{4} - 0.2x^{2} + 0.4x + 3
		/// </summary>
		public static float[] MysteryFunc(params float[] inputs)
		{
			float x = inputs[0];
			return new float[] { (0.002f * MathF.Pow(x, 4)) - (0.2f * MathF.Pow(x, 2)) + (0.4f * x) + 3 };
		}

		public static void Main(string[] args)
		{
			// Conv net testing
			ConvolutionalNeuralNetworkMono testConvNetAlpha = new ConvolutionalNeuralNetworkMono(1, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 2, 4 }, new[] { NDNodeActivFunc.ReLU, NDNodeActivFunc.ReLU }, new[] { ConvPoolingOp.Max, ConvPoolingOp.Max });

			testConvNetAlpha.InitializeThis();

			float[,] inputMatrix = Random.Shared.NextMatrix(100, 100, true);

			testConvNetAlpha.ComputeResultMono(inputMatrix);
		}

		//public static void Main(string[] args)
		//{
		//	const int initialWeightAmp = 5, testCount = 3;
		//	Random random = new Random();
		//	double t1Avg = 0, t2Avg = 0;
		//	for (int i = 0; i < testCount; i++)
		//	{
		//		DateTime t1Start = DateTime.Now;
		//		float GetRandVal(int x, int y, float v) => (float)random.NextDouble() * (initialWeightAmp * 2) - initialWeightAmp;
		//		float[,] test1 = new float[4000, 4000];
		//		test1.SetEach(GetRandVal);
		//		DateTime t1End = DateTime.Now, t2Start = DateTime.Now;
		//		float[,] test2 = (ZMatrix)random.NextMatrix(4000, 4000) * (initialWeightAmp * 2) - initialWeightAmp;
		//		DateTime t2End = DateTime.Now;

		//		t1Avg += Math.Pow((t1End - t1Start).TotalSeconds, 2);
		//		t2Avg += Math.Pow((t2End - t2Start).TotalSeconds, 2);
		//		//Console.WriteLine($"{test1.Length:n}\t{test2.Length:n}");
		//		//Console.WriteLine($"Test 1:\t{(t1End - t1Start).TotalSeconds:f}s\nTest 2:\t{(t2End - t2Start).TotalSeconds:f}s");
		//	}
		//	Console.WriteLine($"---------------------\nTest 1:\t{t1Avg / testCount:f4}s\nTest 2:\t{t2Avg / testCount:f4}s");
		//	Console.ReadKey();
		//}
	}
}