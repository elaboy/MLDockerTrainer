using CommandLine;
using MLDockerTrainer;
using MLDockerTrainer.Assembler;
using MLDockerTrainer.Datasets;
using MLDockerTrainer.Utils;
using Proteomics.PSM;
using TorchSharp.Modules;

public static class Program
{

    public static void Main(string[] args)
    {
        var parser = new Parser(with => with.HelpWriter = null);
        var result = parser.ParseArguments<Settings>(args);

        result.WithParsed(settings => Run(settings));

    }

    private static void Run(Settings settings)
    {
        var model = settings.ModelToTrain;
        var dataPath = settings.DataPath;
        var trainingSplit = settings.TrainingSplit;
        var validationSplit = settings.ValidationSplit;
        var testSplit = settings.TestSplit;
        var sourceVocabularySize = settings.SourceVocabularySize;
        var targetVocabularySize = settings.TargetVocabularySize;
        var sourceSequenceLength = settings.SourceSequenceLength;
        var targetSequenceLength = settings.TargetSequenceLength;
        var dModel = settings.DModel;
        var N = settings.N;
        var h = settings.H;
        var dropout = settings.Dropout;
        var dFF = settings.DFF;
        var batchSize = settings.BatchSize;
        var epochs = settings.Epochs;
        var learningRate = settings.LearningRate;
        var useLearnignRateScheduler = settings.UseLearnignRateScheduler;
        var learningRateDecay = settings.LearningRateDecay;
        var learningRateDecayStep = settings.LearningRateDecayStep;
        var useEarlyStopping = settings.UseEarlyStopping;
        var earlyStoppingPatience = settings.EarlyStoppingPatience;
        var saveModelEachEpoch = settings.SaveModelEachEpoch;
        var vocabularyFilePath = settings.VocabularyFilePath;

        //Use settings to ensamble the model and the training data //todo: work with model naming
        if (model == "transformer" &&
            sourceVocabularySize is not null &&
            targetVocabularySize is not null &&
            sourceSequenceLength is not null &&
            targetSequenceLength is not null &&
            dModel is not null &&
            N is not null &&
            h is not null &&
            dropout is not null &&
            dFF is not null &&
            vocabularyFilePath is not null)
        {
            //Create the transformer model
            var transformer = RetentionTimeTransformerAssembler.AssembleTransformer(
                sourceVocabularySize.Value, targetVocabularySize.Value,
                sourceSequenceLength.Value, targetSequenceLength.Value, dModel.Value, N.Value,
                h.Value, dropout.Value, dFF.Value);

            //Collect data from files
            var psms = new List<PsmFromTsv>();

            dataPath.ForEach(path =>
            {
                var psmList = Readers.SpectrumMatchTsvReader.ReadPsmTsv(path, out var warnings)
                    .Where(x => x.AmbiguityLevel == "1");
                psms.AddRange(psmList);
            });

            //Get vocabulary
            var vocabulary = TokenKit.GetVocab(vocabularyFilePath);

            //Transform data to tokens
            var dataAsVocabFormat = PsmParser.GetRetentionTimeWithFullSequenceAsTokens(psms);
            var formattedData = TokenKit.Tokenize(dataAsVocabFormat, vocabulary);

            //Split into trianing, validation, and test
            var (trainingData, validationData, testData) =
                TokenKit.TrainValidateTestSplit(formattedData, trainingSplit, validationSplit, testSplit);

            //Make datasets for each partition 
            var trainingDataset = new RetentionTimeDataset(trainingData);
            var validationDataset = new RetentionTimeDataset(validationData);
            var testDataset = new RetentionTimeDataset(testData);

            //Make dataloaders for all datasets 
            var trainingDataLoader = new DataLoader(trainingDataset, batchSize, true, drop_last: true);
            var validationDataLoader = new DataLoader(validationDataset, batchSize, true, drop_last: true);
            var testDataLoader = new DataLoader(testDataset, batchSize, true, drop_last: true);

            //Train the model
            transformer.Train(trainingData);
        }
    }

    private static void TrainModel(Transformer transformer, DataLoader trainingDataLoader,
        DataLoader validationDataLoader, DataLoader testDataLoader, int epochs, double learningRate,
        bool useLearnignRateScheduler, double? learningRateDecay, int? learningRateDecayStep, bool useEarlyStopping,
        int? earlyStoppingPatience, bool saveModelEachEpoch)
    {
        throw new System.NotImplementedException();
    }

    private static void PrintHelp()
    {
        System.Console.WriteLine("This is the help message");
    }


}
