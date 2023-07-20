using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ZUtilLib
{
	public static class GenericUtils
	{
		public delegate TOut PromiseFormat<TOut, TIn>(TIn inputData);

		public static async Task<TOutput> Then<TOutput, TInput>(this Task<TInput> task, PromiseFormat<TOutput, TInput> promise)
		{
			TInput result = await task; // CONTINUE HERE OR SOMETHING JUST MAKE IT WORK
			if (promise != null)
			{
				return promise(result);
			}
			else
				throw new Exception("Missing promise method!");
		}
	}
}
