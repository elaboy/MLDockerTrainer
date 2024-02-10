using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLDockerTrainer.Utils;

namespace Tests.TestUtils
{
    public class TestRetentionTimeCalibrator
    {
        [Test]
        public void TestToCSV()
        {
            var calibrator = new RetentionTimeCalibrator(new List<string>()
            {
                @"D:\MannPeptideResults\GAMG_AllPeptides.psmtsv",
                @"D:\MannPeptideResults\HEK293_AllPeptides.psmtsv",
                @"D:\MannPeptideResults\A549_AllPeptides.psmtsv"
            });

            var csv = calibrator.GetDataAsDataTable();
            
            RetentionTimeCalibrator.ToCSV(csv, @"D:\MannPeptideResults\TestingCalibrator.csv");
        }
    }
}
