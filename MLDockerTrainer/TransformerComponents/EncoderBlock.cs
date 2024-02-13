using MachineLearning.Transformer;
using TorchSharp;
using TorchSharp.Modules;

namespace MachineLearning.TransformerComponents;

public class EncoderBlock : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>
{
    public EncoderBlock(int features, MultiHeadAttentionBlock selfAttentionBlock, FeedForwardBlock feedForwardBlock, double dropout) 
        : base(nameof(EncoderBlock))
    {
        _features = features;
        _selfAttentionBlock = selfAttentionBlock;
        _feedForwardBlock = feedForwardBlock;
        _residualConnection = new ModuleList<ResidualConnection>(new ResidualConnection[]
        { new ResidualConnection(features, torch.nn.Dropout(dropout)),
            new ResidualConnection(features, torch.nn.Dropout(dropout))
        });

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input, torch.Tensor sourceMask)
    {
        Func<torch.Tensor, torch.Tensor> subLayer = x => _selfAttentionBlock.forward(x, x, x, sourceMask);

        input = _residualConnection[0].forward(input, subLayer);
        input = _residualConnection[1].forward(input, _feedForwardBlock.forward);
        return input;
    }

    private MultiHeadAttentionBlock _selfAttentionBlock;
    private FeedForwardBlock _feedForwardBlock;
    private ModuleList<ResidualConnection> _residualConnection;
    private int _features;
}