using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLDockerTrainer.Utils;

namespace Tests.TestUtils
{
    public class TestCalibratedRetentionTimes
    {
        [Test]
        public void TestCalibratedRetentionTimesConstructor()
        {
            var calibrator = new CalibratedRetentionTimes(@"D:\MannPeptideResults\TestingCalibrator.csv");

            int zero = 0;
        }
    }
}
