using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThermoFisher.CommonCore.Data;
using TorchSharp;
using TorchSharp.Modules;

namespace MLDockerTrainer.Models.RetentionTimePredictionModels
{
    public class Simple : torch.nn.Module<torch.Tensor, torch.Tensor>
    {
        public Simple(int inputSize, int hiddenSize, int outputSize) : base(nameof(Simple))
        {
            _linear1 = torch.nn.Linear(inputSize, hiddenSize);
            _linear2 = torch.nn.Linear(hiddenSize, hiddenSize);
            _linear3 = torch.nn.Linear(hiddenSize, hiddenSize);
            _linear4 = torch.nn.Linear(hiddenSize, hiddenSize);
            _linear5 = torch.nn.Linear(hiddenSize, outputSize);
            _relu = torch.nn.ReLU();
            _batchNorm1d1 = torch.nn.BatchNorm1d(hiddenSize);
            _batchNorm1d2 = torch.nn.BatchNorm1d(hiddenSize);
            _batchNorm1d3 = torch.nn.BatchNorm1d(hiddenSize);
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor input)
        {
            input = _linear1.forward(input);
            input = _relu.forward(input);
            input = _linear2.forward(input);
            input = _batchNorm1d1.forward(input);
            input = _relu.forward(input);
            input = _linear3.forward(input);
            input = _batchNorm1d2.forward(input);
            input = _relu.forward(input);
            input = _linear4.forward(input);
            input = _batchNorm1d3.forward(input);
            input = _relu.forward(input);
            input = _linear5.forward(input);
            return input;
        }

        public void Train(Simple model, DataLoader trainingDataLoader, DataLoader validationDataLoader,
            DataLoader testingDataLoader, int epochs, double learningRate,
            bool saveModelAfterEachEpoch = true)
        {

            //device
            var device = torch.cuda.is_available() ? torch.device(DeviceType.CUDA) : torch.device(DeviceType.CPU);

            //Move model to device
            model.to(device);

            //Optimizer 
            var lossFunction = torch.nn.MSELoss().to(device);

            //Loss function
            var optimizer = torch.optim.SGD(model.parameters(), learningRate);

            //Summary Writer
            var writer = torch.utils.tensorboard.SummaryWriter("runs/SimpleModel", createRunName: true);



            //Training loop
            for (int i = 0; i < epochs; i++)
            {
                double loss = 0;
                int correct = 0;
                int total = 0;
                int trainingStep = 0;
                model.train();
                foreach (var batch in trainingDataLoader)
                {
                    var data = batch["Sequence"].to(device);
                    var labels = batch["Target"].to(device);

                    //Forward pass
                    var predictions = model.forward(data).to(device);

                    //Calculate loss
                    var batchLoss = lossFunction.forward(predictions, labels);
                    loss += batchLoss.item<float>();

                    //Backward pass
                    optimizer.zero_grad();
                    batchLoss.backward();

                    //Update weights
                    optimizer.step();

                    //Write loss to tensorboard
                    writer.add_scalar("loss/trainingLoss", batchLoss.item<float>(), trainingStep);
                    trainingStep++;
                }

                //Calculate validation accuracy
                int validationCorrect = 0;
                int validationTotal = 0;
                int validationStep = 0;
                model.eval();
                foreach (var batch in validationDataLoader)
                {
                    var data = batch["Sequence"].to(device);
                    var labels = batch["Target"].to(device);

                    //Forward pass
                    var predictions = model.forward(data);

                    //Calculate loss
                    var batchLoss = lossFunction.forward(predictions, labels);
                    loss += batchLoss.item<float>();

                    //Backward pass
                    optimizer.zero_grad();
                    batchLoss.backward();

                    //Update weights
                    optimizer.step();

                    //Write loss to tensorboard
                    writer.add_scalar("loss/validationLoss", batchLoss.item<float>(), validationStep);
                    validationStep++;
                }

                model.save(@$"C:\Users\elabo\Documents\model_{i}");
            }
        }
        private Linear _linear1;
        private Linear _linear2;
        private Linear _linear3;
        private Linear _linear4;
        private Linear _linear5;
        private ReLU _relu;
        private BatchNorm1d _batchNorm1d1;
        private BatchNorm1d _batchNorm1d2;
        private BatchNorm1d _batchNorm1d3;
    }
}
