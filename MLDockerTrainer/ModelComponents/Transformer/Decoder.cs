using TorchSharp;
using TorchSharp.Modules;

namespace MLDockerTrainer.ModelComponents.Transformer;

public class Decoder : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>
{
    public Decoder(ModuleList<DecoderBlock> layers) : base(nameof(Decoder))
    {
        _layers = layers;

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input, torch.Tensor encoderOutput, torch.Tensor sourceMask,
        torch.Tensor targetMask)
    {
        foreach (var layer in _layers)
            input = layer.forward(input, encoderOutput, sourceMask, targetMask);

        return _norm.forward(input);
    }

    private ModuleList<DecoderBlock> _layers;
    private LayerNormalization _norm = new LayerNormalization();
}