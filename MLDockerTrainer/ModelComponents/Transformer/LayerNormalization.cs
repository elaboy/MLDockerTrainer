using TorchSharp;
using TorchSharp.Modules;

namespace MLDockerTrainer.ModelComponents.Transformer;

public class LayerNormalization : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    public LayerNormalization(double eps = 1e-6) : base(nameof(LayerNormalization))
    {
        _eps = eps;
        _alpha = torch.nn.Parameter(torch.ones(1)); //multiplied
        _beta = torch.nn.Parameter(torch.zeros(1)); //added
        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input)
    {
        var mean = input.mean(new long[]{0}, true);
        
        var std = input.std(0, true);
        
        var norm = (input - mean) / (std + _eps);
        
        return _alpha * norm + _beta;
    }

    private double _eps;
    private Parameter _alpha;
    private Parameter _beta;
}