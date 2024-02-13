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
                @"C:\Users\elabo\Documents\MannPeptideResults\HEK293_AllPeptides.psmtsv",
                @"C:\Users\elabo\Documents\MannPeptideResults\Hela_AllPeptides.psmtsv",
                @"C:\Users\elabo\Documents\MannPeptideResults\HepG2AllPeptides.psmtsv",
                @"C:\Users\elabo\Documents\MannPeptideResults\Jurkat_AllPeptides.psmtsv",
                @"C:\Users\elabo\Documents\MannPeptideResults\A549_AllPeptides.psmtsv",
                @"C:\Users\elabo\Documents\MannPeptideResults\GAMG_AllPeptides.psmtsv"
            });

            var csv = calibrator.GetDataAsDataTable();

            RetentionTimeCalibrator.ToCSV(csv, @"C:\Users\elabo\Documents\MannPeptideResults\TestingCalibrator.csv");
        }

        [Test]
        public void TestCheckingVarianceFromRtvsAvg()
        {
            var calibrator = new RetentionTimeCalibrator(new List<string>()
            {
                @"C:\Users\elabo\Documents\MannPeptideResults\HEK293_AllPeptides.psmtsv"
            });

            var csv = calibrator.GetDataAsDataTable();

            RetentionTimeCalibrator.ToCSV(csv, @"C:\Users\elabo\Documents\MannPeptideResults\TestingCalibratorTestingVariance.csv");
        }
    }
}