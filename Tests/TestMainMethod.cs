using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests
{
    public class TestMainMethod
    {
        [Test]
        public void TestMainMethodRun()
        {
            Program.Main(new string[] { @"C:\Users\elabo\Documents\GitHub\MLDockerTrainer\settings.txt" });
        }
    }
}
