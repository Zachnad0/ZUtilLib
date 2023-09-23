using System;
using System.Collections.Generic;
using System.Linq;
using ZUtilLib.ZAI;
using ZUtilLib.ZAI.Saving;
using System.Threading.Tasks;
using ZUtilLib.ZAI.Training;

namespace LIBRARYTESTING
{
	public class Program
	{
		public static async Task Main(string[] args)
		{
			Console.WriteLine("Start");

			NeuralNetwork testNet = new NeuralNetwork(3, 3, 5, 2, NDNodeActivFunc.ReLU);
			testNet.InitializeThis();
			testNet.SetupOutputs("alpha", "beta", "gamma");
			(string NodeName, float Value)[] result = testNet.PerformCalculations(("in1", 0.2f), ("in2", 0.3f), ("in3", 0.4f));
			NeuralNetwork secondTestNet = new NeuralNetwork(3, 3, 5, 2, NDNodeActivFunc.ReLU);
			secondTestNet.InitializeThis(testNet, 1, 0.1f, true);
			secondTestNet.SetupOutputs("delta", "epsilon", "zeta");
			var result2 = secondTestNet.PerformCalculations(("in1", 0.2f), ("in2", 0.3f), ("in3", 0.4f));

			PackagedNeuralNetwork pNN1 = new PackagedNeuralNetwork(testNet);
			PackagedNeuralNetwork pNN2 = new PackagedNeuralNetwork(secondTestNet);

			NeuralNetwork cn1, cn2;
			cn1 = new NeuralNetwork(pNN1);
			cn2 = new NeuralNetwork(pNN2);

			NeuralNetTrainingOptions nNTO = new NeuralNetTrainingOptions(100, 5, 80, 1, 1, 5, 3, 0.6f, 0.7f, NDNodeActivFunc.SoftPlus, false, true, 2.5f, -10, 10, generations: 10000);
			NeuralNetwork[] topNets = await NeuralNetTraining.TrainNetworksOnFunc(MysteryFunc, nNTO);

			// Check effectiveness of top networks
			for (int n = 0; n < topNets.Length; n++)
			{
				float diff = 0;
				for (int i = 0; i < 100; i++)
				{
					float x = (10 - -10) / 100f * i + -10f;
					float y1 = topNets[n].PerformCalculations(x)[0];
					float y2 = MysteryFunc(x)[0];
					diff += MathF.Abs(y1 - y2);
				}
				//diff /= 100;
				Console.WriteLine($"Avg diffs: #{n}: {diff}");
			}

			Console.WriteLine("Finished");
		}

		/// <summary>
		/// f(x) = 0.002x^{4} - 0.2x^{2} + 0.4x + 3
		/// </summary>
		public static float[] MysteryFunc(params float[] inputs)
		{
			float x = inputs[0];
			return new float[] { (0.002f * MathF.Pow(x, 4)) - (0.2f * MathF.Pow(x, 2)) + (0.4f * x) + 3 };
		}
	}
}