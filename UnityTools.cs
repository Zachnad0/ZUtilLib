using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace ZUtilLib
{
	public static class UnityTools
	{
		public static Quaternion ToQuaternion(this Vector3 v)
		{
			return Quaternion.LookRotation(v, Vector3.Scale(v, Vector3.up));
		}
		public static Vector3 ToVector3(this Quaternion q)
		{
			return Vector3.Normalize(q * Vector3.forward);
		}
	}
}
