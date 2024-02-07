using Easy.Common.Extensions;

namespace MLDockerTrainer.Utils
{
    public static class DatasetKit
    {
        public static (List<List<int>>, List<List<int>>, List<List<int>>)
            TrainValidateTestSplit(List<List<int>> listOfTokenId, double trainingFraction = 0.8,
                double testingFraction = 0.1, double validationFraction = 0.1)
        {
            var randomizedData = listOfTokenId.Randomize().ToList();

            var trainingSet = randomizedData
                .Take((int)(randomizedData.Count * (1 - validationFraction)))
                .ToList();

            var validationSet = randomizedData
                .Skip((int)(randomizedData.Count * (1 - validationFraction)))
                .Take((int)(randomizedData.Count * (1 - testingFraction)))
                .ToList();

            var testSet = randomizedData
                .Skip((int)(randomizedData.Count * (1 - validationFraction)))
                .ToList();

            return (trainingSet, validationSet, testSet);
        }
    }
}
