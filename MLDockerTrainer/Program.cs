using CommandLine;
using MachineLearning.RetentionTimePredictionModels;
using MLDockerTrainer;
using MLDockerTrainer.Assembler;
using MLDockerTrainer.Datasets;
using MLDockerTrainer.ModelComponents.Transformer;
using MLDockerTrainer.Utils;
using Proteomics.PSM;
using TorchSharp;

public static class Program
{
    public static void Main(string[] args)
    {
        var parser = new Parser(with => with.HelpWriter = null);
        //var result = parser.ParseArguments<string>(args);
        Console.WriteLine(args[0]);

        var settings = new Settings(args[0]);

        foreach (var prop in settings.GetType().GetProperties())
        {
            Console.WriteLine($"{prop.Name}: {prop.GetValue(settings)}");
        }

        Console.WriteLine("Running...");
        Run(settings);
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
        var useLearnignRateScheduler = settings.UseLearningRateScheduler;
        var learningRateDecay = settings.LearningRateDecay;
        var learningRateDecayStep = settings.LearningRateDecayStep;
        var useEarlyStopping = settings.UseEarlyStopping;
        var earlyStoppingPatience = settings.EarlyStoppingPatience;
        var saveModelEachEpoch = settings.SaveModelEachEpoch;
        var vocabularyFilePath = settings.VocabularyFilePath;

        //Use settings to ensamble the model and the training data //todo: work with model naming, maybe an enum?
        if (model == "AARTN-Transformer" &&
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
            var transformer = AARTN.AssembleTranformer(
                sourceVocabularySize.Value, targetVocabularySize.Value,
                sourceSequenceLength.Value, targetSequenceLength.Value, dModel.Value, N.Value,
                h.Value, dropout.Value, dFF.Value);

            ////Collect data from files
            //var psms = new List<PsmFromTsv>();

            //dataPath.ForEach(path =>
            //{
            //    var psmList = Readers.SpectrumMatchTsvReader.ReadPsmTsv(path, out var warnings)
            //        .Where(x => x.AmbiguityLevel == "1");
            //    psms.AddRange(psmList);
            //});

            //load calibrated retention times
            CalibratedRetentionTimes calibratedRetentionTimes = new CalibratedRetentionTimes(settings.DataPath.First());

            //Get vocabulary
            var vocabulary = TokenKit.GetVocab(vocabularyFilePath);

            //Transform data to tokens
            var formattedFullSequences =
                TokenKit.TokenizeFullSequence(calibratedRetentionTimes);

            var dataAsVocabFormat = TokenKit.Tokenize(formattedFullSequences, vocabulary);

            //Split into trianing, validation, and test
            var (trainingData, validationData, testData) =
                TokenKit.TrainValidateTestSplit(dataAsVocabFormat, trainingSplit, validationSplit,
                    testSplit);

            //Make datasets for each partition 
            var trainingDataset = new RetentionTimeDataset(trainingData);
            var validationDataset = new RetentionTimeDataset(validationData);
            var testDataset = new RetentionTimeDataset(testData);

            //Make dataloaders for all datasets 
            var trainingDataLoader = new TorchSharp.Modules.DataLoader(trainingDataset, batchSize, true, drop_last: true);
            var validationDataLoader = new TorchSharp.Modules.DataLoader(validationDataset, batchSize, true, drop_last: true);
            var testDataLoader = new TorchSharp.Modules.DataLoader(testDataset, batchSize, true, drop_last: true);

            //Train the model
            AARTN.Train(transformer, trainingDataLoader, validationDataLoader, testDataLoader, epochs, learningRate, 
                useLearnignRateScheduler, learningRateDecay, learningRateDecayStep, useEarlyStopping, 
                earlyStoppingPatience, saveModelEachEpoch);
        }
    }

    private static void TrainModel(Transformer transformer, TorchSharp.Modules.DataLoader trainingDataLoader,
        TorchSharp.Modules.DataLoader validationDataLoader, TorchSharp.Modules.DataLoader testDataLoader, int epochs, double learningRate,
        bool useLearnignRateScheduler = false, double? learningRateDecay = null, int? learningRateDecayStep = null, bool useEarlyStopping = false,
        int? earlyStoppingPatience = null, bool saveModelEachEpoch = true)
    {
        //Define the device to use for training
        var device = torch.cuda.is_available() ? torch.device(DeviceType.CUDA) : torch.device(DeviceType.CPU);

        //Move the model to the device
        transformer.to(device);

        //Create the optimizer
        var optimizer = torch.optim.Adam(transformer.parameters(), learningRate);

        //Loss Function 
        var lossFunction = torch.nn.CrossEntropyLoss(ignore_index: 0).to(device);

        //Scheduler check
        torch.optim.lr_scheduler.LRScheduler? scheduler = null;

        if (useLearnignRateScheduler)
        {
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, learningRateDecayStep.Value, learningRateDecay.Value);
        }

        //Summary Writer
        var writer = torch.utils.tensorboard.SummaryWriter(@$"runs/{transformer.GetName()}", createRunName: true);

        //Model Stats
        var trainingLossTracker = new List<List<float>>();
        var validationLossTracker = new List<List<float>>();
        var trainingSteps = 0;
        var validatingSteps = 0;

        //Learning loop
        for (int i = 0; i < epochs; i++)
        {
            transformer.train();
            List<float> epochLossLog = new();

            foreach (var batch in trainingDataLoader)
            {

                var encoderInput = batch["EncoderInput"].to(device);
                var decoderInput = batch["DecoderInput"].to(device);
                var encoderMask = batch["EncoderInputMask"].to(device);
                var decoderMask = batch["DecoderInputMask"].to(device);
                var label = batch["Label"].to(device);

                //Run tensors through the transformer
                var encoderOutput = transformer.Encode(encoderInput, encoderMask).to(device);
                var decoderOutput = transformer.Decode(encoderOutput, encoderMask,
                    decoderInput, decoderMask).to(device);
                var projectionOutput = transformer.Project(decoderOutput).to(device);


                var prediction = torch.FloatTensor(projectionOutput).to(device);
                var target = torch.LongTensor(label).to(device);

                var loss = lossFunction.forward(prediction, target);

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                epochLossLog.Add(loss.item<float>());

                //Write to tensorboard
                writer.add_scalar("Loss/Training", loss.item<float>(), trainingSteps);
                trainingSteps++;
            }

            trainingLossTracker.Add(epochLossLog);

            //Validate the model and quantify the loss
            transformer.eval();
            List<float> validationLossLog = new();

            foreach (var batch in validationDataLoader)
            {
                var encoderInput = batch["EncoderInput"].to(device);
                var decoderInput = batch["DecoderInput"].to(device);
                var encoderMask = batch["EncoderInputMask"].to(device);
                var decoderMask = batch["DecoderInputMask"].to(device);
                var label = batch["Label"].to(device);

                //Run tensors through the transformer
                var encoderOutput = transformer.Encode(encoderInput, encoderMask).to(device);
                var decoderOutput = transformer.Decode(encoderOutput, encoderMask,
                                       decoderInput, decoderMask).to(device);
                var projectionOutput = transformer.Project(decoderOutput).to(device);

                var prediction = torch.FloatTensor(projectionOutput).to(device);
                var target = torch.LongTensor(label).to(device);

                var loss = lossFunction.forward(prediction, target);

                validationLossLog.Add(loss.item<float>());

                //Write to tensorboard
                writer.add_scalar("Loss/Validation", loss.item<float>(), validatingSteps);
                validatingSteps++;
            }

            validationLossTracker.Add(validationLossLog);

            //save model if required
            if (saveModelEachEpoch)
            {
                transformer.save(@$"checkpoint/epoch_{epochs}_{transformer.GetName()}.dat");
            }
        }

        //Test the model using the testing dataset (final validation step)
        transformer.eval();
        List<float> testLossLog = new();
        var testingSteps = 0;
        foreach (var batch in validationDataLoader)
        {
            var encoderInput = batch["EncoderInput"].to(device);
            var decoderInput = batch["DecoderInput"].to(device);
            var encoderMask = batch["EncoderInputMask"].to(device);
            var decoderMask = batch["DecoderInputMask"].to(device);
            var label = batch["Label"].to(device);

            //Run tensors through the transformer
            var encoderOutput = transformer.Encode(encoderInput, encoderMask).to(device);
            var decoderOutput = transformer.Decode(encoderOutput, encoderMask,
                                                  decoderInput, decoderMask).to(device);
            var projectionOutput = transformer.Project(decoderOutput).to(device);

            var prediction = torch.FloatTensor(projectionOutput).to(device);
            var target = torch.LongTensor(label).to(device);

            var loss = lossFunction.forward(prediction, target);

            testLossLog.Add(loss.item<float>());

            //Write to tensorboard
            writer.add_scalar("Loss/Test", loss.item<float>(), testingSteps);
            testingSteps++;
        }

        transformer.save(@$"trainedModelWeights_{transformer.GetName()}.dat");
    }

    private static void PrintHelp()
    {
        System.Console.WriteLine("This is the help message");
    }


}
