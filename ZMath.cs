using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Runtime.InteropServices;
using System.Data.Common;
using System.Runtime.CompilerServices;

namespace ZUtilLib.ZMath // OOAC (Object Oriented Algebraic Calculator) system
{
	public class Expression : IExpressionable
	{
		public IExpressionable[] Terms { get; private set; }

		public Expression(params IExpressionable[] terms) => Terms = terms.Where(e => e != this && e != null).ToArray();

		public static bool TryParse(string rawEquation, out Expression expression)
		{
			// WIP
			expression = null;
			return false;
		}

		public float SubstituteValues(params (Variables, float)[] subs)
		{
			float output = 0;
			foreach (IExpressionable term in Terms)
				output += term.SubstituteValues(subs);
			return output;
		}

		public IExpressionable Differentiate(Variables diffBy)
		{
			// Iterate through, retrieve differentiated copies
			IExpressionable[] expressionables = new IExpressionable[Terms.Length];
			for (int i = 0; i < Terms.Length; i++)
			{
				expressionables[i] = Terms[i].Differentiate(diffBy);
			}
			if (expressionables.Length > 0)
				return new Expression(expressionables);
			return null;
		}
	}

	public class Term : IExpressionable
	{
		public IExpressionable Exponent { get; private set; }
		public Variables Variable { get; private set; }

		public Term(Variables variable, IExpressionable exponent, float? exponentF = null)
		{
			Variable = variable;
			if (exponentF != null)
				Exponent = (PlainValue)exponentF.Value;
			else
				Exponent = exponent;
		}

		public float SubstituteValues(params (Variables, float)[] subs)
		{
			IEnumerable<(Variables, float)> varSubs = subs.Where(s => s.Item1 == Variable);
			if (varSubs.Count() != 1) // Ensure my variable is subbed exactly
				throw new MissingVarSubException(Variable, this);
			float varSub = varSubs.First().Item2;

			float exp = 0;
			if (Exponent != null)
				exp = Exponent.SubstituteValues(subs);

			return (float)Math.Pow(varSub, exp);
		}

		public IExpressionable Differentiate(Variables diffBy)
		{
			if (Variable != diffBy)
				return null;

			// Determine this thing diffed via chain rule
			IExpressionable newExponent = new Expression(Exponent, new PlainValue(-1));
			IExpressionable diffThis = new MultiplyGroup(Exponent, new Term(Variable, newExponent, null));

			return diffThis;
		}
	}

	public class MultiplyGroup : IExpressionable
	{
		public List<IExpressionable> Members { get; private set; }

		public MultiplyGroup(params IExpressionable[] members) => Members = members.Where(m => m != this && m != null).ToList();

		public float SubstituteValues(params (Variables, float)[] subs)
		{
			float output = 1;
			foreach (IExpressionable member in Members)
				output *= member.SubstituteValues(subs);
			return output;
		}

		public IExpressionable Differentiate(Variables diffBy)
		{
			if (Members.Count == 1)
				return Members[0].Differentiate(diffBy);

			if (Members.Count > 1)
			{ // Product rule current and next, then make current the result of that
				IExpressionable current = Members[0];
				for (int i = 1; i < Members.Count; i++) // FORBIDDEN "int i = 1" SITUATION???
				{
					current = new Expression(new MultiplyGroup(current, Members[i].Differentiate(diffBy)), new MultiplyGroup(Members[i], current.Differentiate(diffBy)));
				}
				return current;
			}

			return null;
		}
	}

	public class PlainValue : IExpressionable
	{
		public float Value { get; private set; }
		public PlainValue(float value) { Value = value; }

		public float SubstituteValues(params (Variables, float)[] subs) => Value;

		public IExpressionable Differentiate(Variables diffBy) => new PlainValue(0);

		public static implicit operator PlainValue(float f) => new PlainValue(f);
	}

	public enum Variables
	{
		a, b, c, d, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z,
	}

	public interface IExpressionable
	{
		float SubstituteValues(params (Variables, float)[] subs);
		IExpressionable Differentiate(Variables diffBy);
		public static Expression operator +(IExpressionable left, IExpressionable right) => new Expression(left, right);
		public static MultiplyGroup operator *(IExpressionable left, IExpressionable right) => new MultiplyGroup(left, right);
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
			Expression exp = new Expression(
				(PlainValue)0.2f * (IExpressionable)new Term(Variables.x, null, 2),
				(PlainValue)(-8) * (IExpressionable)new Term(Variables.x, null, 1),
				(PlainValue)4
			);
			Console.WriteLine(exp.SubstituteValues((Variables.x, 3))); // -18.2

			IExpressionable diff = exp.Differentiate(Variables.x);
			Console.WriteLine(diff.SubstituteValues((Variables.x, 3))); // -6.8
		}
	}
}
