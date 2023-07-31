using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using System.ComponentModel;
using System.Net;
using System.Runtime.CompilerServices;
using System.Text;

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
	}
}
