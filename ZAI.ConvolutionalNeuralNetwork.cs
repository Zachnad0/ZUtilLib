using System;
using System.Numerics;

namespace ZUtilLib.ZAI
{
	public class ConvolutionalNeuralNetworkMono
	{
		private FilterPoolNodeMono[][] FilterAndPoolNodes { get; set; }

		public ConvolutionalNeuralNetworkMono(int kernelWH, int[] convAndPoolLayerHeights)
		{
			FilterAndPoolNodes = new FilterPoolNodeMono[convAndPoolLayerHeights.Length][];
			for (int i = 0; i < convAndPoolLayerHeights.Length; i++)
				FilterAndPoolNodes[i] = new FilterPoolNodeMono[convAndPoolLayerHeights[i]];
		}

		public ConvNetworkResult ComputeResultMono(float[,] monoPixelGrid)
		{
			monoPixelGrid = monoPixelGrid.NormalizeMatrix(false);

			return default;
		}
	}

	internal class FilterPoolNodeMono
	{
		public float[][,] _cachedData = null;

		public (FilterPoolNodeMono Node, float[,] Kernel)[] NodeLinkKernels { get; private set; }
		public float Bias { get; private set; }
		private GraphStuff.GraphEquation _activationFunc;

		public FilterPoolNodeMono((FilterPoolNodeMono Node, float[,] Kernel)[] nodeLinkKernels, float bias, GraphStuff.GraphEquation activationFunc)
		{
			NodeLinkKernels = nodeLinkKernels;
			Bias = bias;
			_activationFunc = activationFunc;
		}

		/// <returns>float[channel][x, y]</returns>
		public float[][,] CalculateData() // Width = (w - f)/s + 1
		{
			if (_cachedData != null) return _cachedData;

			float[][,] outChannels = new float[NodeLinkKernels.Length][,];
			for (int nodeN = 0; nodeN < NodeLinkKernels.Length; nodeN++) // For nodes
			{
				var channels = NodeLinkKernels[nodeN].Node.CalculateData();
				for (int chanN = 0; chanN < channels.Length; chanN++) // CONTINUE HERE with this insanity
				{
					float[,] pixelData = channels[chanN];

					// Scan pixelData and get the sum of the components of the dot between the kernel and current focus
					int horizSteps = pixelData.GetLength(0) - NodeLinkKernels[nodeN].Kernel.GetLength(0) + 1;
					int vertSteps = pixelData.GetLength(1) - NodeLinkKernels[nodeN].Kernel.GetLength(1) + 1;
					outChannels[nodeN] = new float[horizSteps, vertSteps];

					// Convolute current channel via kernel
					for (int vp = pixelData.GetLength(1) - 1; vp >= NodeLinkKernels[nodeN].Kernel.GetLength(1) - 1; vp--) // Down pixelData Y
					{
						for (int hp = 0; hp < horizSteps; hp++) // Right pixelData X
						{
							float dotSum = 0;
							// Calculate sum of dot components
							for (int x = 0; x < NodeLinkKernels[nodeN].Kernel.GetLength(0); x++) // Right filter X
							{
								for (int y = 0; y < NodeLinkKernels[nodeN].Kernel.GetLength(1); y++) // Down filter Y
								{
									dotSum += NodeLinkKernels[nodeN].Kernel[x, y] * pixelData[hp + x, vp - y];
								}
							}
							outChannels[nodeN][hp, vp - NodeLinkKernels[nodeN].Kernel.GetLength(1) + 1] = _activationFunc(dotSum + Bias);
						}
					}

					// Pool via max of sample size
					// CONTINUE HERE with pool sampling
					outChannels[nodeN] = outChannels[nodeN].NormalizeMatrix(false); // Maybe?
				}
			}
			return outChannels;
		}

		public FilterPoolNodeMono Clone()
		{
			throw new NotImplementedException();
		}
	}

	public readonly struct ConvNetworkResult
	{
		public int DecidedType { get; }
		public float CertaintyOfType { get; }
		public Vector2 TopLeftBound { get; }
		public Vector2 BottomRightBound { get; }

		internal ConvNetworkResult(int decidedType, float certaintyOfType, Vector2 topLeftBound, Vector2 bottomRightBound)
		{
			DecidedType = decidedType;
			CertaintyOfType = certaintyOfType;
			TopLeftBound = topLeftBound;
			BottomRightBound = bottomRightBound;
		}
	}
}
