using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

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
			return default;
		}
	}

	internal class FilterPoolNodeMono
	{
		private float[,] _cachedData = null;

		public float[,] Kernel { get; private set; }
		private GraphStuff.GraphEquation _activationFunc;

		public FilterPoolNodeMono(float[,] kernel, GraphStuff.GraphEquation activationFunc)
		{
			Kernel = kernel;
			_activationFunc = activationFunc;
		}

		public float[,] CalculateData(float[,] pixelData) // Calculate the resulting matrix
		{
			if (_cachedData != null) return _cachedData;

			// Scan pixelData and get the sum of the components of the dot between the kernel and current focus
			int horizSteps = pixelData.GetLength(0) - Kernel.GetLength(0) + 1;
			int vertSteps = pixelData.GetLength(1) - Kernel.GetLength(1) + 1;
			float[,] outputData = new float[horizSteps, vertSteps];

			// Convolute via filter
			for (int vp = pixelData.GetLength(1) - 1; vp >= Kernel.GetLength(1) - 1; vp--) // Down and right
			{
				for (int hp = 0; hp < horizSteps; hp++)
				{
					float dotSum = 0;
					// Calculate sum of dot components
					for (int x = 0; x < Kernel.GetLength(0); x++) // Right and down
					{
						for (int y = 0; y < Kernel.GetLength(1); y++)
						{
							dotSum += Kernel[x, y] * pixelData[hp + x, vp - y];
						}
					}
					outputData[hp, vp - Kernel.GetLength(1) + 1] = _activationFunc(dotSum);
				}
			}

			// Pool via max of sample size
			// CONTINUE HERE with pool sampling

			return outputData;
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
