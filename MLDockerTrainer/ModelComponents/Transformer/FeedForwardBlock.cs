using TorchSharp;
using TorchSharp.Modules;

namespace MLDockerTrainer.ModelComponents.Transformer;

public class FeedForwardBlock : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    public FeedForwardBlock(int dModel, int dFF, double dropout = 0.1) : base(nameof(FeedForwardBlock))
    {
        _linear1 = torch.nn.Linear(dModel, dFF); //W1 and B1
        _linear2 = torch.nn.Linear(dFF, dModel); //W2 and B2
        _dropout = torch.nn.Dropout(dropout);

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input)
    {
        // (batchSize, sequenceLength, dModel) to (batchSize, sequenceLength, dFF) to (batchSize, sequenceLength, dModel)
        return _linear2.forward(_dropout.forward(torch.nn.functional.relu(_linear1.forward(input))));
    }

    private Linear _linear1;
    private Linear _linear2;
    private Dropout _dropout;
}