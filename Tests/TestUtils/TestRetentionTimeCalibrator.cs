using MLDockerTrainer.Utils;

namespace Tests.TestUtils
{
    public class TestRetentionTimeCalibrator
    {
        [Test]
        public void TestToCSV()
        {
            //Get inside the MannPeptidesResults folder and get all ides.psmtsv files paths into the paths list
            List<string> paths = new();
            foreach (var file in Directory.GetFiles(@"C:\Users\elabo\Documents\MannPeptideResults"))
            {
                if (file.Contains("AllPeptides.psmtsv") && !file.Contains(".txt"))
                {
                    paths.Add(file);
                    break;
                }
            }

            var calibrator = new RetentionTimeCalibrator(paths);

            var csv = calibrator.GetDataAsDataTable();

            RetentionTimeCalibrator.ToCSV(csv, @"C:\Users\elabo\Documents\MannPeptideResults\CalibratorTestingMultipleFilesSmallFiltered.csv");
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