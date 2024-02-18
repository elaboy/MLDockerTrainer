using TorchSharp;
using TorchSharp.Modules;

namespace MLDockerTrainer.ModelComponents.Transformer;

public class ProjectionLayer : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    public ProjectionLayer(int dModel, int vocabSize) : base(nameof(ProjectionLayer))
    {
        _projectionLayer = torch.nn.Linear(dModel, 100);
        _prijectionLayer2 = torch.nn.Linear(100, 10);
        _prijectionLayer3 = torch.nn.Linear(10, vocabSize);

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input)
    {
        //(Batch, SequenceLength, dModel) to (Batch, SequenceLength, vocabSize)
        input = input.squeeze(0);
        input = _projectionLayer.forward(input);
        input = _prijectionLayer2.forward(input);
        return _prijectionLayer3.forward(input);
    }

    private Linear _projectionLayer;
    private Linear _prijectionLayer2;
    private Linear _prijectionLayer3;

}