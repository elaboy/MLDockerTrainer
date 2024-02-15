using System.Diagnostics;
using Tensorboard;
using TorchSharp;
using TorchSharp.Modules;
using Decoder = MLDockerTrainer.ModelComponents.Transformer.Decoder;
using DecoderBlock = MLDockerTrainer.ModelComponents.Transformer.DecoderBlock;
using Encoder = MLDockerTrainer.ModelComponents.Transformer.Encoder;
using EncoderBlock = MLDockerTrainer.ModelComponents.Transformer.EncoderBlock;
using FeedForwardBlock = MLDockerTrainer.ModelComponents.Transformer.FeedForwardBlock;
using InputEmbeddings = MLDockerTrainer.ModelComponents.Transformer.InputEmbeddings;
using MultiHeadAttentionBlock = MLDockerTrainer.ModelComponents.Transformer.MultiHeadAttentionBlock;
using PositionalEncoder = MLDockerTrainer.ModelComponents.Transformer.PositionalEncoder;
using ProjectionLayer = MLDockerTrainer.ModelComponents.Transformer.ProjectionLayer;
using Transformer = MLDockerTrainer.ModelComponents.Transformer.Transformer;

namespace MachineLearning.RetentionTimePredictionModels;
public static class AARTN
{
    public static Transformer AssembleTranformer(int sourceVocabSize, int targetVocabSize,
        int sourceSequenceLength, int targetSequenceLength, int dModel = 512, int N = 6, int h = 8,
        double dropout = 0.1, int dFF = 2048)
    {
        //Create the embedding layers
        var sourceEmbedding = new InputEmbeddings(dModel, sourceVocabSize);
        var targetEmbedding = new InputEmbeddings(dModel, targetVocabSize);

        //Create the positional encoding layers
        var sourcePosition = new PositionalEncoder(dModel, sourceSequenceLength, dropout);
        var targetPosition = new PositionalEncoder(dModel, targetSequenceLength, dropout);

        //Create the encoder blocks
        var encoderBlocks = new List<EncoderBlock>();
        for (int i = 0; i < N; i++)
        {
            var decoderSelfAttentionBlock = new MultiHeadAttentionBlock(dModel, h, dropout);
            var feedForwardBlock = new FeedForwardBlock(dModel, dFF, dropout);
            var encoderBlock = new EncoderBlock(dModel, decoderSelfAttentionBlock, feedForwardBlock, dropout);
            encoderBlocks.Add(encoderBlock);
        }

        //Create the decoder blocks
        var decoderBlocks = new List<DecoderBlock>();
        for (int i = 0; i < N; i++)
        {
            var decoderSelfAttentionBlock = new MultiHeadAttentionBlock(dModel, h, dropout);
            var encoderDecoderAttentionBlock = new MultiHeadAttentionBlock(dModel, h, dropout);
            var feedForwardBlock = new FeedForwardBlock(dModel, dFF, dropout);
            var decoderBlock = new DecoderBlock(dModel, decoderSelfAttentionBlock,
                encoderDecoderAttentionBlock,
                                   feedForwardBlock, dropout);
            decoderBlocks.Add(decoderBlock);
        }

        //Create the enconder and the decorder 
        var encoder = new Encoder(torch.nn.ModuleList<EncoderBlock>(encoderBlocks
            .Select(x => x).ToArray()));
        var decoder = new Decoder(torch.nn.ModuleList<DecoderBlock>(decoderBlocks
            .Select(x => x).ToArray()));

        //Create the projection layer
        var projectionLayer = new ProjectionLayer(dModel, targetVocabSize);

        //Create the transformer
        var transformer = new MLDockerTrainer.ModelComponents.Transformer.Transformer(encoder, decoder,
            sourceEmbedding, targetEmbedding, sourcePosition,
                           targetPosition, projectionLayer);

        return transformer;
    }

    public static void Train(Transformer transformer, DataLoader trainingDataLoader, DataLoader validationDataLoader,
        DataLoader testingDataLoader, int epochs, double learningRate, bool useLearningRateScheduler,
        double? learningRateDecay, int? learningRateDecayStep, bool useEarlyStopping, int? earlyStopPatience,
        bool saveModelAfterEachEpoch)
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

        if (useLearningRateScheduler)
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

                //Debug.WriteLine(prediction.ToString(TensorStringStyle.Julia));
                //Debug.WriteLine(prediction[torch.TensorIndex.Colon, torch.TensorIndex.Colon,
                //        torch.TensorIndex.Slice(0, 22)]
                //    .ToString(TensorStringStyle.Julia));

                //Debug.WriteLine(target.ToString(TensorStringStyle.Julia));

                var loss = lossFunction.forward(
                    prediction[torch.TensorIndex.Colon, torch.TensorIndex.Colon,
                        torch.TensorIndex.Slice(0, 22)], target);

                Debug.WriteLine("trainingLoss: " + loss.ToString(TensorStringStyle.Julia));
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
                Debug.WriteLine("validationLoss: " + loss.ToString(TensorStringStyle.Julia));

                validationLossLog.Add(loss.item<float>());

                //Write to tensorboard
                writer.add_scalar("Loss/Validation", loss.item<float>(), validatingSteps);
                validatingSteps++;
            }

            validationLossTracker.Add(validationLossLog);

            //save model if required
            if (saveModelAfterEachEpoch)
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

}
