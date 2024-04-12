using System;

namespace ZUtilLib.ZAI
{
	/// <summary>
	/// Types of neural data node activation functions.
	/// </summary>
	public enum NDNodeActivFunc
	{
		ReLU, LeakyReLU, Sigmoid, SoftPlus, HyperbolicTangent, ELU, Swish, GELU,
	}

	/// <summary>
	/// Types of pooling operations for convolutional neural networks.
	/// </summary>
	public enum ConvPoolingOp
	{
		Max, Average
	}

	public static class Equations
	{
		/// <summary>
		/// Uses the current graph equation to get y from x.
		/// </summary>
		/// <param name="x">X value.</param>
		/// <returns>y value.</returns>
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
				NDNodeActivFunc.LeakyReLU => LeakyReLUEquation,
				NDNodeActivFunc.Sigmoid => SigmoidEquation,
				NDNodeActivFunc.SoftPlus => SoftPlusEquation,
				NDNodeActivFunc.HyperbolicTangent => HyperbolicTangentEquation,
				NDNodeActivFunc.ELU => ELUEquation,
				NDNodeActivFunc.Swish => SwishEquation,
				NDNodeActivFunc.GELU => GELUEquation,
				_ => throw new NotImplementedException(),
			};
		}

		public static float ReLUEquation(float x) => MathF.Max(0, x);
		public static float LeakyReLUEquation(float x) => MathF.Max(0.1f * x, x);
		public static float SigmoidEquation(float x) => 1 / (1 + MathF.Exp(-x));
		public static float SoftPlusEquation(float x)
		{
			float y = MathF.Log(1 + MathF.Exp(x));
			return y switch
			{
				float.PositiveInfinity => float.MaxValue,
				float.NegativeInfinity => float.MinValue,
				_ => y,
			};
		}
		public static float HyperbolicTangentEquation(float x) => MathF.Tanh(x);
		public static float ELUEquation(float x) => x >= 0 ? x : (MathF.Exp(x) - 1);
		public static float SwishEquation(float x) => x / (1 + MathF.Exp(-x));
		public static float GELUEquation(float x) => 0.5f * x * (1 + MathF.Tanh(MathF.Sqrt(2 / MathF.PI) * (x + 0.044715f * MathF.Pow(x, 3))));
	}
	public static class MatrixOperations
	{
		/// <summary>
		/// Uses the provided sample matrix to give a float value.
		/// </summary>
		/// <param name="matrix">The sample to be processed.</param>
		/// <returns>Resulting value.</returns>
		public delegate float ConvOp(float[,] matrix);

		/// <summary>
		/// This is used to obtain the pooling operation based on the type.
		/// </summary>
		/// <param name="operationType">Convolutional pooling node operation type.</param>
		/// <returns>The delegate of the corresponding method.</returns>
		public static ConvOp GetOperationFromType(ConvPoolingOp operationType)
		{
			return operationType switch
			{
				ConvPoolingOp.Max => MaxOperation,
				ConvPoolingOp.Average => AverageOperation,
				_ => throw new NotImplementedException(),
			};
		}

		public static float MaxOperation(float[,] matrix)
		{
			float max = float.MinValue;
			foreach (float val in matrix)
				max = MathF.Max(max, val);
			return max;
		}
		public static float AverageOperation(float[,] matrix)
		{
			float sum = 0;
			foreach (float val in matrix)
				sum += val;
			return sum / (matrix.GetLength(0) * matrix.GetLength(1));
		}
	}
}