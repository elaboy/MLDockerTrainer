using TorchSharp;
using TorchSharp.Modules;

namespace MLDockerTrainer.ModelComponents.Transformer;

public class ResidualConnection : torch.nn.Module<torch.Tensor, Func<torch.Tensor, torch.Tensor>, torch.Tensor>
{
    public ResidualConnection(int features, Dropout dropout) : base(nameof(ResidualConnection))
    {
        _dropout = dropout;
        _norm = new LayerNormalization(features);
        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input, Func<torch.Tensor, torch.Tensor> subLayer)
    {
        return input + _dropout.forward(subLayer.Invoke(_norm.forward(input)));
    }

    private Dropout _dropout;
    private LayerNormalization _norm = new LayerNormalization();
}