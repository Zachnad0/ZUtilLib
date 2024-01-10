using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using System.Text;

namespace ZUtilLib
{
	/// <summary>
	/// This static class is full of static utility methods, with most of them being extension.
	/// </summary>
	public static class ZUtils
	{
		/// <summary>
		/// Loads the given JSON file and deserializes it.
		/// </summary>
		/// <typeparam name="T">Type to be parsed.</typeparam>
		/// <param name="path">Path of file</param>
		/// <returns></returns>
		public static async Task<(bool, T)> LoadJSONFromFile<T>(string path, JsonSerializerOptions options)
		{
			if (File.Exists(path))
			{
				using (FileStream fileStream = File.OpenRead(path))
				{
					try
					{
						T result = await JsonSerializer.DeserializeAsync<T>(fileStream, options);
						return (true, result);
					}
					catch
					{
					}
				}

			}

			return (false, default);
		}

		/// <summary>
		/// Loads the given JSON file, then overwrites it with the input data.
		/// </summary>
		/// <param name="thing">The "thing" to be serialized</param>
		/// <param name="path">Path of file</param>
		/// <param name="doNotOverwrite">Set to true if instead the data should be appended</param>
		/// <returns>If the operation was successful</returns>
		public static async Task<bool> WriteJSONToFile<T>(T thing, string path, JsonSerializerOptions options, bool doNotOverwrite = false)
		{
			if (File.Exists(path))
			{
				using (FileStream fileStream = File.OpenWrite(path))
				{
					if (!doNotOverwrite)
					{
						await JsonSerializer.SerializeAsync(fileStream, thing, options);
					}
					else
					{
						string output = JsonSerializer.Serialize(thing, options);
						await fileStream.WriteAsync(Encoding.UTF8.GetBytes(output), (int)fileStream.Length, int.MaxValue);
					}
					return true;
				}
			}

			return false;
		}

		/// <summary>
		/// It does exactly what you think, in both possible ways.
		/// </summary>
		/// <param name="str">String to be filtered.</param>
		/// <param name="removeNotIsolate">If true, return a string <b>without</b> the numbers. If false, return a string <b>with</b> only the numbers.</param>
		/// <param name="allowDecimal">If true, allows decimal points in the string.</param>
		/// <returns>A new string of whatever option you chose for <paramref name="removeNotIsolate"/>.</returns>
		public static string FilterNumbers(this string str, bool removeNotIsolate, bool allowDecimal = false)
		{
			char[] allowedChars = new char[] { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
			if (!removeNotIsolate)
				return new string(str.Where(x => allowedChars.Contains(x) || (allowDecimal && x == '.')).ToArray());
			List<char> output = str.ToList();
			output.RemoveAll(x => allowedChars.Contains(x));
			return new string(output.ToArray());
		}

		/// <summary>
		/// Converts a matrix of 0-255 value bytes into 0-1 value floats.
		/// </summary>
		/// <param name="byteMatrix">The matrix to be converted</param>
		/// <returns>A new matrix of floats.</returns>
		public static float[,] ToFloatMatrix(this byte[,] byteMatrix)
		{
			float[,] output = new float[byteMatrix.GetLength(0), byteMatrix.GetLength(1)];
			for (int x = 0; x < output.GetLength(0); x++)
				for (int y = 0; y < output.GetLength(1); y++)
					output[x, y] = byteMatrix[x, y] / 255f;
			return output;
		}

		/// <summary>
		/// Normalizes the matrix so that all values are between 1 and (-1 if <paramref name="negativeFloor"/> is true, otherwise 0).
		/// </summary>
		/// <param name="matrix">The matrix to be normalized</param>
		/// <param name="negativeFloor">If true, will normalize values between 1 and -1, instead of 1 and 0.</param>
		/// <returns>The normalized matrix.</returns>
		public static float[,] NormalizeMatrix(this float[,] matrix, bool negativeFloor)
		{
			int width = matrix.GetLength(0), height = matrix.GetLength(1);
			float[,] normMatrix = new float[width, height];
			float maxVal = float.NegativeInfinity, minVal = float.PositiveInfinity;
			foreach (float f in matrix)
			{
				if (f > maxVal)
					maxVal = f;
				else if (f < minVal)
					minVal = f;
			}

			// Normalization
			for (int x = 0; x < width; x++)
			{
				for (int y = 0; y < height; y++)
				{
					// x = (x - min) / (max - min)
					normMatrix[x, y] = (matrix[x, y] - minVal) / (maxVal - minVal);
					if (negativeFloor)
						normMatrix[x, y] = 2 * normMatrix[x, y] - 1;
				}
			}

			return normMatrix;
		}

		/// <summary>
		/// Normalizes the matrix so that all values are between 1 and (-1 if <paramref name="negativeFloor"/> is true, otherwise 0).
		/// </summary>
		/// <param name="array">The array to be normalized</param>
		/// <param name="negativeFloor">If true, will normalize values between 1 and -1, instead of 1 and 0.</param>
		/// <returns>The normalized array.</returns>
		public static float[] NormalizeArray(this float[] array, bool negativeFloor)
		{
			float maxVal = float.NegativeInfinity, minVal = float.PositiveInfinity;
			float[] normArray = new float[array.Length];
			foreach (float f in array)
			{
				if (f > maxVal)
					maxVal = f;
				else if (f < minVal)
					minVal = f;
			}
			for (int i = 0; i < array.Length; i++)
			{
				normArray[i] = (array[i] - minVal) / (maxVal - minVal);
				if (negativeFloor)
					normArray[i] = 2 * normArray[i] - 1;
			}

			return normArray;
		}

		/// <summary>
		/// Generates a matrix of size <paramref name="height"/> and <paramref name="width"/>, consisting of random values between (0, or -1 if <paramref name="negativeFloor"/> is true) and 1.
		/// </summary>
		/// <param name="random">Current System.Random instance.</param>
		/// <param name="height">Width of the matrix.</param>
		/// <param name="width">Height of the matrix.</param>
		/// <param name="negativeFloor">If true, will make the minimum value -1 instead of 0.</param>
		/// <returns>An unnormalized matrix of size <paramref name="width"/> by <paramref name="height"/>.</returns>
		public static float[,] NextMatrix(this Random random, int width, int height, bool negativeFloor = false)
		{
			float[,] newMatrix = new float[width, height];
			for (int x = 0; x < width; x++)
			{
				for (int y = 0; y < height; y++)
				{
					newMatrix[x, y] = (float)random.NextDouble();
					if (negativeFloor)

						newMatrix[x, y] = newMatrix[x, y] * 2 - 1;
				}
			}
			return newMatrix;
		}

		/// <summary>
		/// Generates an array of size <paramref name="length"/>, consisting of random values between (0, or -1 if <paramref name="negativeFloor"/> is true) and 1.
		/// </summary>
		/// <param name="random">Current System.Random instance.</param>
		/// <param name="length">Length of the array.</param>
		/// <param name="negativeFloor">If true, will make the minimum value -1 instead of 0.</param>
		/// <returns>An unnormalized array of size <paramref name="length"/>.</returns>
		public static float[] NextArray(this Random random, int length, bool negativeFloor = false)
		{
			float[] newArray = new float[length];
			for (int i = 0; i < length; i++)
			{
				newArray[i] = (float)random.NextDouble();
				if (negativeFloor)
					newArray[i] = newArray[i] * 2 - 1;
			}
			return newArray;
		}

		/// <summary>
		/// Iterates through the matrix and runs <paramref name="action"/> for each value, passing in the current x, y, and <typeparamref name="T"/>.
		/// </summary>
		/// <param name="matrix">The matrix.</param>
		/// <param name="action">The method that takes in x, y and the current <typeparamref name="T"/>.</param>
		public static T[,] Foreach<T>(this T[,] matrix, Action<int, int, T> action)
		{
			int w = matrix.GetLength(0), h = matrix.GetLength(1);
			for (int x = 0; x < w; x++)
				for (int y = 0; y < h; y++)
					action(x, y, matrix[x, y]);
			return matrix;
		}
		/// <summary>
		/// Iterates through the matrix and runs <paramref name="action"/> for each value, passing in the current x, y, and <typeparamref name="T"/>.
		/// </summary>
		/// <param name="matrix">The matrix.</param>
		/// <param name="action">The method that takes in x, y and the current <typeparamref name="T"/>.</param>
		public static T[][] Foreach<T>(this T[][] matrix, Action<int, int, T> action)
		{
			int w = matrix.Length;
			for (int x = 0; x < w; x++)
			{
				int h = matrix[x].Length;
				for (int y = 0; y < h; y++)
					action(x, y, matrix[x][y]);
			}
			return matrix;
		}
		/// <summary>
		/// Iterates through the array and runs <paramref name="action"/> for each value, passing in the current i and <typeparamref name="T"/>.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <param name="action">The method that takes in i and the current <typeparamref name="T"/>.</param>
		public static T[] Foreach<T>(this T[] array, Action<int, T> action)
		{
			int w = array.Length;
			for (int i = 0; i < w; i++)
				action(i, array[i]);
			return array;
		}

		/// <summary>
		/// Iterates through the matrix and runs <paramref name="func"/> for each value, passing in the current x, y, and <typeparamref name="T"/>, then setting the value of the matrix at that location to be the result.
		/// </summary>
		/// <param name="matrix">The matrix.</param>
		/// <param name="func">The method that takes in x, y and the current <typeparamref name="T"/>, and returns the new current value.</param>
		public static T[,] SetEach<T>(this T[,] matrix, Func<int, int, T, T> func)
		{
			int w = matrix.GetLength(0), h = matrix.GetLength(1);
			for (int x = 0; x < w; x++)
				for (int y = 0; y < h; y++)
					matrix[x, y] = func(x, y, matrix[x, y]);
			return matrix;
		}
		/// <summary>
		/// Iterates through the matrix and runs <paramref name="func"/> for each value, passing in the current x, y, and <typeparamref name="T"/>, then setting the value of the matrix at that location to be the result.
		/// </summary>
		/// <param name="matrix">The matrix.</param>
		/// <param name="func">The method that takes in x, y and the current <typeparamref name="T"/>, and returns the new current value.</param>
		public static T[][] SetEach<T>(this T[][] matrix, Func<int, int, T, T> func)
		{
			int w = matrix.Length;
			for (int x = 0; x < w; x++)
			{
				int h = matrix[x].Length;
				for (int y = 0; y < h; y++)
					matrix[x][y] = func(x, y, matrix[x][y]);
			}
			return matrix;
		}
		/// <summary>
		/// Iterates through the array and runs <paramref name="func"/> for each value, passing in the current iterator and <typeparamref name="T"/>, then setting the value of the array at that location to be the result.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <param name="func">The method that takes in 'i' and the current <typeparamref name="T"/>, and returns the new current value.</param>
		public static T[] SetEach<T>(this T[] array, Func<int, T, T> func)
		{
			int l = array.Length;
			for (int i = 0; i < l; i++)
					array[i] = func(i, array[i]);
			return array;
		}

		/// <summary>
		/// Finds the lowest common multiple of all of the values within an array.
		/// </summary>
		/// <param name="values">An array of the values to find the LCM for.</param>
		/// <returns>The lowest common multiple.</returns>
		public static double LCM(params double[] values)
		{
			var sortedVals = values.OrderByDescending(v => v).ToArray();

			double lCM = sortedVals[0];
			for (int i = 1; i < sortedVals.Length; i++)
			{
				double preLCM = lCM;
				while (lCM % sortedVals[i] != 0)
					lCM += preLCM;
			}

			return lCM;
		}

		/// <summary>
		/// Converts an array of lines of strings into a jagged-type character matrix.
		/// </summary>
		/// <returns>A completely new 2D jagged character array with no references to the <paramref name="lines"/>.</returns>
		public static char[][] LinesToCharMatrix(this string[] lines)
		{
			int mWidth = lines[0].Length, mHeight = lines.Length;
			char[][] outMatrix = new char[mWidth][];
			for (int x = 0; x < mWidth; x++)
			{
				outMatrix[x] = new char[mHeight];
				for (int y = 0; y < mHeight; y++)
				{
					outMatrix[x][y] = lines[y][x];
				}
			}

			return outMatrix;
		}

		/// <summary>
		/// Generates a customizable, nice-looking, human readable representation of a 2D array, as a string.
		/// </summary>
		/// <param name="horizSeperator">This string is inserted between each array element of each row.</param>
		/// <param name="vertSeperator">This string is inserted after each row of elements.</param>
		/// <returns></returns>
		public static string ToReadableString<T>(this T[][] matrix, string horizSeperator = "\t", string vertSeperator = "\n")
		{
			string outputString = "";
			int mWidth = matrix.Length, mHeight = matrix[0].Length; ;
			for (int y = 0; y < mHeight; y++)
			{
				for (int x = 0; x < mWidth; x++)
					outputString += $"{(x != 0 ? horizSeperator : "")}{matrix[x][y]}";
				outputString += y != mWidth - 1 ? vertSeperator : "";
			}

			return outputString;
		}

		/// <summary>
		/// Rotates the matrix 90 degrees if <paramref name="clockwise"/>, otherwise counter-clockwise.
		/// </summary>
		/// <param name="clockwise">True if matrix should be rotated clockwise, false for anti-clockwise.</param>
		/// <returns>A new matrix containing the elements in rotated positions depending on <paramref name="clockwise"/>.</returns>
		public static T[][] RotateMatrix<T>(this T[][] matrix, bool clockwise)
		{
			int origMHeight = matrix[0].Length, origMWidth = matrix.Length;
			T[][] outMatrix = new T[origMHeight][];
			for (int i = 0; i < origMHeight; i++)
				outMatrix[i] = new T[origMWidth];

			for (int origMY = 0; origMY < origMHeight; origMY++)
			{
				for (int origMX = 0; origMX < origMWidth; origMX++)
				{
					if (clockwise)
						outMatrix[origMHeight - 1 - origMY][origMX] = matrix[origMX][origMY];
					else
						outMatrix[origMY][origMWidth - 1 - origMX] = matrix[origMX][origMY];
				}
			}

			return outMatrix;
		}
		/// <summary>
		/// Rotates the matrix 90 degrees if <paramref name="clockwise"/>, otherwise counter-clockwise.
		/// </summary>
		/// <param name="clockwise">True if matrix should be rotated clockwise, false for anti-clockwise.</param>
		/// <returns>A new matrix containing the elements in rotated positions depending on <paramref name="clockwise"/>.</returns>
		public static T[,] RotateMatrix<T>(this T[,] matrix, bool clockwise)
		{
			int origMHeight = matrix.GetLength(1), origMWidth = matrix.GetLength(0);
			T[,] outMatrix = new T[origMHeight, origMWidth];

			for (int origMY = 0; origMY < origMHeight; origMY++)
			{
				for (int origMX = 0; origMX < origMWidth; origMX++)
				{
					if (clockwise)
						outMatrix[origMHeight - 1 - origMY, origMX] = matrix[origMX, origMY];
					else
						outMatrix[origMY, origMWidth - 1 - origMX] = matrix[origMX, origMY];
				}
			}

			return outMatrix;
		}

		/// <summary>
		/// Compares the two matrices' dimensions and elements to determine their identicalness.
		/// </summary>
		/// <returns>Whether all of their practical properties are identical.</returns>
		public static bool Identical<T>(this T[][] matrix, T[][] otherMatrix)
		{
			if (matrix is null || otherMatrix is null || matrix.Length != otherMatrix.Length || matrix[0].Length != otherMatrix[0].Length)
				return false;

			int mWidth = matrix.Length, mHeight = matrix[0].Length;
			for (int y = 0; y < mHeight; y++)
				for (int x = 0; x < mWidth; x++)
					if (!matrix[x][y].Equals(otherMatrix[x][y]))
						return false;

			return true;
		}
		/// <summary>
		/// Compares the two matrices' dimensions and elements to determine their identicalness.
		/// </summary>
		/// <returns>Whether all of their practical properties are identical.</returns>
		public static bool Identical<T>(this T[,] matrix, T[,] otherMatrix)
		{
			if (matrix is null || otherMatrix is null || matrix.GetLength(0) != otherMatrix.GetLength(0) || matrix.GetLength(1) != otherMatrix.GetLength(1))
				return false;

			int mWidth = matrix.GetLength(0), mHeight = matrix.GetLength(1);
			for (int y = 0; y < mHeight; y++)
				for (int x = 0; x < mWidth; x++)
					if (!matrix[x, y].Equals(otherMatrix[x, y]))
						return false;

			return true;
		}

		/// <summary>
		/// Converts a <u>rectangular</u> jagged matrix into a non-jagged rectangular matrix.
		/// </summary>
		/// <returns>A 2D non-jagged matrix of the values of the provided jagged matrix.</returns>
		public static T[,] ToNonJaggedMatrix<T>(this T[][] matrix)
		{
			if (matrix is null || matrix.Any(c => c.Length != matrix[0].Length))
				throw new ArgumentException();

			T[,] newMatrix = new T[matrix.Length, matrix[0].Length];
			int mWidth = newMatrix.GetLength(0), mHeight = newMatrix.GetLength(1);
			for (int y = 0; y < mHeight; y++)
				for (int x = 0; x < mWidth; x++)
					newMatrix[x, y] = matrix[x][y];

			return newMatrix;
		}

		/// <summary>
		/// Converts a non-jagged matrix into a jagged matrix.
		/// </summary>
		/// <returns>A 2D jagged matrix of the values of the provided matrix.</returns>
		public static T[][] ToJaggedMatrix<T>(this T[,] matrix)
		{
			if (matrix == null)
				throw new ArgumentNullException();

			int mWidth = matrix.GetLength(0), mHeight = matrix.GetLength(1);
			T[][] outMatrix = new T[mWidth][];
			for (int x = 0; x < mWidth; x++)
			{
				outMatrix[x] = new T[mHeight];
				for (int y = 0; y < mHeight; y++)
					outMatrix[x][y] = matrix[x, y];
			}

			return outMatrix;
		}

		/// <summary>
		/// <i>Properly</i> clones this matrix without the worry of the usual mistakes.
		/// </summary>
		public static T[][] CloneJaggedMatrix<T>(this T[][] matrix)
		{
			int mWidth = matrix.Length;
			T[][] outMatrix = new T[matrix.Length][];
			for (int x = 0; x < mWidth; x++)
				outMatrix[x] = (T[])matrix[x].Clone();
			return outMatrix;
		}

		public static class GreekAlphabet
		{
			public const char alpha = 'α', Alpha = 'Α', beta = 'β', Beta = 'Β', gamma = 'γ', Gamma = 'Γ', delta = 'δ', Delta = 'Δ', epsilon = 'ε', Epsilon = 'Ε', zeta = 'ζ', Zeta = 'Ζ', eta = 'η', Eta = 'Η', theta = 'θ', Theta = 'Θ', iota = 'ι', Iota = 'Ι', kappa = 'κ', Kappa = 'Κ', lambda = 'λ', Lambda = 'Λ', mu = 'μ', Mu = 'Μ', nu = 'ν', Nu = 'Ν', xi = 'ξ', Xi = 'Ξ', omicron = 'ο', Omicron = 'Ο', pi = 'π', Pi = 'Π', rho = 'ρ', Rho = 'Ρ', sigma = 'σ', Sigma = 'Σ', tau = 'τ', Tau = 'Τ', upsilon = 'υ', Upsilon = 'Υ', phi = 'φ', Phi = 'Φ', chi = 'χ', Chi = 'Χ', psi = 'ψ', Psi = 'Ψ', omega = 'ω', Omega = 'Ω';
			public static readonly char[] LowerCaseLetters = { alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambda, mu, nu, xi, omicron, pi, rho, sigma, tau, upsilon, phi, chi, psi, omega };
			public static readonly char[] UpperCaseLetters = { Alpha, Beta, Gamma, Delta, Epsilon, Zeta, Eta, Theta, Iota, Kappa, Lambda, Mu, Nu, Xi, Omicron, Pi, Rho, Sigma, Tau, Upsilon, Phi, Chi, Psi, Omega };
		}
	}
}
