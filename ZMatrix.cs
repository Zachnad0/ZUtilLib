namespace ZUtilLib
{
	/// <summary>
	/// While not intended to be used directly, this class allows for easily using operators to do matrix things. Casting to this class allows for adding and multiplying matrices and stuff, so it can be useful. <u>Check the source code for more information.</u>
	/// </summary>
	public class ZMatrix
	{
		private float[,] _matrix;
		private ZMatrix(float[,] matrix) => _matrix = matrix;

		// Implicit conversions
		public static implicit operator ZMatrix(float[,] matrix) => new ZMatrix(matrix.Clone() as float[,]);
		public static implicit operator ZMatrix(float[][] jaggedMatrix) => new ZMatrix(jaggedMatrix.ToNonJaggedMatrix()); // Convert from jagged
		public static implicit operator float[,](ZMatrix zMatrix) => zMatrix._matrix.Clone() as float[,];

		#region OperatorOverrides
		// Operators

		public static ZMatrix operator -(ZMatrix m) => m * -1;
		public static ZMatrix operator *(ZMatrix m1, float multiplier)
		{
			int w = m1._matrix.GetLength(0), h = m1._matrix.GetLength(1);
			for (int x = 0; x < w; x++)
				for (int y = 0; y < h; y++)
					m1._matrix[x, y] *= multiplier;
			return m1;
		}
		public static ZMatrix operator +(ZMatrix m1, float additional)
		{
			int w = m1._matrix.GetLength(0), h = m1._matrix.GetLength(1);
			for (int x = 0; x < w; x++)
				for (int y = 0; y < h; y++)
					m1._matrix[x, y] += additional;
			return m1;
		}
		public static ZMatrix operator -(ZMatrix m1, float subtractional)
		{
			int w = m1._matrix.GetLength(0), h = m1._matrix.GetLength(1);
			for (int x = 0; x < w; x++)
				for (int y = 0; y < h; y++)
					m1._matrix[x, y] -= subtractional;
			return m1;
		}
		/// <summary>
		/// Multiplies each of the components with one another. Requires two identically sized arrays.
		/// </summary>
		public static ZMatrix operator *(ZMatrix m1, ZMatrix m2)
		{
			int w, h;
			if ((w = m1._matrix.GetLength(0)) != m2._matrix.GetLength(0) || (h = m1._matrix.GetLength(1)) != m2._matrix.GetLength(1))
				throw new System.Exception("ZMatrix * ZMatrix Operator Critical Error: Matrices are of differing dimensions.");
			ZMatrix result = new ZMatrix(new float[w, h]);
			for (int x = 0; x < w; x++)
				for (int y = 0; y < h; y++)
					result._matrix[x, y] = m1._matrix[x, y] * m2._matrix[x, y];
			return result;
		}
		public static bool operator ==(ZMatrix m1, ZMatrix m2)
		{ // The equals method is left unmodified for instance checking
			int mWidth = m1._matrix.GetLength(0), mHeight = m1._matrix.GetLength(1);
			if (mWidth != m2._matrix.GetLength(0) || mHeight != m2._matrix.GetLength(1))
				return false;
			for (int x = 0; x < mWidth; x++)
				for (int y = 0; y < mHeight; y++)
					if (m1._matrix[x, y] != m2._matrix[x, y])
						return false;
			return true;
		}
		public static bool operator !=(ZMatrix m1, ZMatrix m2) => !(m1 == m2); // Bruh

		#endregion
		#region Methods
		// Methods

		public static float SumOfDots(ZMatrix m1, ZMatrix m2)
		{
			int w, h;
			if ((w = m1._matrix.GetLength(0)) != m2._matrix.GetLength(0) || (h = m1._matrix.GetLength(1)) != m2._matrix.GetLength(1))
				throw new System.Exception("ZMatrix.SumOfDots Critical Error: Matrices are of differing dimensions.");
			float dotSum = 0;
			for (int x = 0; x < w; x++)
				for (int y = 0; y < h; y++)
					dotSum += m1._matrix[x, y] * m2._matrix[x, y];
			return dotSum;
		}

		/// <summary>
		/// Rotates the matrix 90 degrees if <paramref name="clockwise"/>, otherwise counter-clockwise.
		/// </summary>
		/// <param name="clockwise">True if matrix should be rotated clockwise, false for anti-clockwise.</param>
		public void RotateBy90Deg(bool clockwise) => _matrix = _matrix.RotateMatrix(clockwise);

		#endregion
	}
}
