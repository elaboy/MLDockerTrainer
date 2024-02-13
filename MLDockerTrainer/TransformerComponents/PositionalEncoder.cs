using MathNet.Numerics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace MachineLearning.TransformerComponents;

public class PositionalEncoder : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    public PositionalEncoder(int dModel, int sequenceLenth, double dropout) : base(nameof(PositionalEncoder))
    {
        //Create vector of shape (sequenceLenth, dModel)
        var positionalEncoding = torch.zeros(sequenceLenth, dModel);
        //Create tensor of shape (sequenceLenth, 1)
        var position = torch.arange(0, sequenceLenth, torch.float32).unsqueeze(1);
        //Division term
        var divTerm = 
            torch.exp(torch.arange(0, dModel, 2) * (-Math.Log(10000.0) / dModel));

        //Apply the sin to even indices and cos to odd indices

        positionalEncoding[torch.TensorIndex.Colon, torch.TensorIndex.Slice(0,null, 2)] = 
            torch.sin(position * divTerm);
        positionalEncoding[torch.TensorIndex.Colon, torch.TensorIndex.Slice(1,null, 2)] = 
            torch.cos(position * divTerm);

        _positionalEncoding = positionalEncoding.unsqueeze(0);/*.to(DeviceType.CUDA)*/; //becomes (1, sequenceLenth, dModel)
        //register_buffer("positionalEncoding", _positionalEncoding);
    }

    public override torch.Tensor forward(torch.Tensor input)
    {
        input = input + _positionalEncoding.requires_grad_(false);
        return _dropout.forward(input);
    }

    private Dropout _dropout = torch.nn.Dropout(0.1);
    private torch.Tensor _positionalEncoding;
}