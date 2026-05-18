using System;
using System.Collections.Generic;
using System.Linq;

namespace Betlang.Fixtures
{
    public sealed class Program
    {
        public static void Main(string[] args)
        {
            var names = new List<string> { "Ada", "Grace", "Linus" };
            foreach (var name in names.Where(value => value.Length > 0))
            {
                Console.WriteLine($"Hello, {name}");
            }
        }
    }
}
