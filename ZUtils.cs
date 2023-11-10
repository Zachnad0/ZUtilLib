using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using System.Text;
using System.Runtime.CompilerServices;

namespace ZUtilLib
{
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

	}

	public static class GreekAlphabet
	{
		public const char alpha = 'α', Alpha = 'Α', beta = 'β', Beta = 'Β', gamma = 'γ', Gamma = 'Γ', delta = 'δ', Delta = 'Δ', epsilon = 'ε', Epsilon = 'Ε', zeta = 'ζ', Zeta = 'Ζ', eta = 'η', Eta = 'Η', theta = 'θ', Theta = 'Θ', iota = 'ι', Iota = 'Ι', kappa = 'κ', Kappa = 'Κ', lambda = 'λ', Lambda = 'Λ', mu = 'μ', Mu = 'Μ', nu = 'ν', Nu = 'Ν', xi = 'ξ', Xi = 'Ξ', omicron = 'ο', Omicron = 'Ο', pi = 'π', Pi = 'Π', rho = 'ρ', Rho = 'Ρ', sigma = 'σ', Sigma = 'Σ', tau = 'τ', Tau = 'Τ', upsilon = 'υ', Upsilon = 'Υ', phi = 'φ', Phi = 'Φ', chi = 'χ', Chi = 'Χ', psi = 'ψ', Psi = 'Ψ', omega = 'ω', Omega = 'Ω';
		public static readonly char[] LowerCaseLetters = { alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambda, mu, nu, xi, omicron, pi, rho, sigma, tau, upsilon, phi, chi, psi, omega };
		public static readonly char[] UpperCaseLetters = { Alpha, Beta, Gamma, Delta, Epsilon, Zeta, Eta, Theta, Iota, Kappa, Lambda, Mu, Nu, Xi, Omicron, Pi, Rho, Sigma, Tau, Upsilon, Phi, Chi, Psi, Omega };
	}
}
