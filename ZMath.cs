using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace ZUtilLib.ZMath // OOAC (Object Oriented Algebraic Calculator) system
{
	public class Expression : IExpressionable
	{
		public IExpressionable[] Terms { get; private set; }

		public Expression(IExpressionable[] terms) => Terms = terms.Where(e => e != this).ToArray();

		public static bool TryParse(string rawEquation, out Expression expression)
		{
			// WIP
			expression = null;
			return false;
		}

		public float SubstituteValue(params (Variables, float)[] subs)
		{
			float output = 0;
			foreach (IExpressionable term in Terms)
				output += term.SubstituteValue(subs);
			return output;
		}
	}

	public class Term : IExpressionable
	{
		public IExpressionable Coefficient { get; private set; }
		public IExpressionable Exponent { get; private set; }
		public float? CoefficientF { get; private set; }
		public float? ExponentF { get; private set; }
		public Variables Variable { get; private set; }

		public Term(IExpressionable coefficient, float? coefficientF, Variables variable, IExpressionable exponent, float? exponentF)
		{
			Variable = variable;
			if (coefficient != null)
				Coefficient = coefficient;
			else
				CoefficientF = coefficientF ?? 1;
			if (exponent != null)
				Exponent = exponent;
			else
				ExponentF = exponentF ?? 1;
		}

		public float SubstituteValue(params (Variables, float)[] subs)
		{
			IEnumerable<(Variables, float)> varSubs = subs.Where(s => s.Item1 == Variable);
			if (varSubs.Count() != 1) // Ensure my variable is subbed exactly
				throw new MissingVarSubException(Variable, this);
			float varSub = varSubs.First().Item2;

			float coeff = 0;
			if (Coefficient != null)
				coeff = Coefficient.SubstituteValue(subs);
			else
				coeff = CoefficientF.Value;

			float exp = 0;
			if (Exponent != null)
				exp = Exponent.SubstituteValue(subs);
			else
				exp = ExponentF.Value;

			return coeff * (float)Math.Pow(varSub, exp);
		}
	}

	public class PlainValue : IExpressionable
	{
		public float Value { get; private set; }
		public PlainValue(float value) { Value = value; }
		public float SubstituteValue(params (Variables, float)[] subs) => Value;
	}

	public enum Variables
	{
		a, b, c, d, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z,
	}

	public interface IExpressionable
	{
		float SubstituteValue(params (Variables, float)[] subs);
	}

	public class MissingVarSubException : Exception
	{
		public MissingVarSubException(Variables missingVar, IExpressionable expr) : base($"Missing {missingVar} in {expr} during substitution!") { }
	}

	public static class TEST
	{
		/// <summary>
		/// Kinda sus
		/// </summary>
		public static void Test()
		{
			Expression exp = new Expression(new IExpressionable[]
			{
				new Term(null, 0.2f, Variables.x, null, 2),
				new Term(null, -8, Variables.x, null, 1),
				new PlainValue(4),
			});
			Console.WriteLine(exp.SubstituteValue((Variables.x, 3)));
		}
	}
}
