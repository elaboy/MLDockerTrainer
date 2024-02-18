using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLDockerTrainer.Utils;
using TorchSharp;
using MLDockerTrainer.Datasets;

namespace Tests
{
    public class TestSimple
    {
        [Test]
        public void TestSimpleRun()
        {
            //load calibrated retention times
            CalibratedRetentionTimes calibratedRetentionTimes =
                new CalibratedRetentionTimes(
                    @"C:\Users\elabo\Documents\GitHub\MLDockerTrainer\CalibratorTestingMultipleFiles.csv");

            //Get vocabulary
            var vocabulary =
                TokenKit.GetVocab(
                    @"C:\Users\elabo\Documents\GitHub\MLDockerTrainer\VocabularyForTransformerUnimod_V4.csv");

            //Transform data to tokens
            var formattedFullSequences =
                TokenKit.TokenizeFullSequence(calibratedRetentionTimes);

            var dataAsVocabFormat = TokenKit.Tokenize(formattedFullSequences, vocabulary);

            //Split into trianing, validation, and test
            var (trainingData, validationData, testData) =
                TokenKit.TrainValidateTestSplit(dataAsVocabFormat, 0.8, 0.1, 0.1);

            //Make datasets for each partition 
            var trainingDataset = new SimpleDataset(trainingData);
            var validationDataset = new SimpleDataset(validationData);
            var testDataset = new SimpleDataset(testData);

            //Make dataloaders for all datasets 
            var trainingDataLoader = new TorchSharp.Modules.DataLoader(trainingDataset, 128, true, drop_last: true);
            var validationDataLoader = new TorchSharp.Modules.DataLoader(validationDataset, 128, true, drop_last: true);
            var testDataLoader = new TorchSharp.Modules.DataLoader(testDataset, 128, true, drop_last: true);

            //Make model
            var model = new MLDockerTrainer.Models.RetentionTimePredictionModels.Simple(100, 1000, 1);

            //Train model
            model.Train(model, trainingDataLoader, validationDataLoader, testDataLoader, 100, 0.0001, false);
        }
    }
}
