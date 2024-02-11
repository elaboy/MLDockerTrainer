using MLDockerTrainer.Utils;

namespace Tests.TestUtils
{
    public class TestTokenKit
    {
        [Test]
        public void TestTokenKitConstructor()
        {
            var calibrator = new CalibratedRetentionTimes(@"D:\MannPeptideResults\TestingCalibrator.csv");
            var tokenizedCalibrator = TokenKit.TokenizeRetentionTimeWithFullSequence(calibrator);

            int zero = 0;
        }
    }
}
