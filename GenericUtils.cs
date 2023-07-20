using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ZUtilLib
{
	public static class GenericUtils
	{
		public delegate void PromiseFormat();

		public static async void Then(this Task task, PromiseFormat promise)
		{
			await task;
			if (promise != null)
				promise();
			else
				throw new Exception("Then missing promise method!");
		}
	}
}
