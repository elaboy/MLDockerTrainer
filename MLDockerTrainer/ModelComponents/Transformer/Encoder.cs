using TorchSharp;
using TorchSharp.Modules;

namespace MLDockerTrainer.ModelComponents.Transformer;

public class Encoder : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>
{
    public Encoder(ModuleList<EncoderBlock> layers) : base(nameof(Encoder))
    {
        _layers = layers;

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input, torch.Tensor mask)
    {
        foreach (var layer in _layers)
            input = layer.forward(input, mask);

        return _norm.forward(input);
    }


    private ModuleList<EncoderBlock> _layers;
    private LayerNormalization _norm = new LayerNormalization();
}