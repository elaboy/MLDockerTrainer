using TorchSharp;
using TorchSharp.Modules;

namespace MLDockerTrainer.ModelComponents.Transformer;

public class DecoderBlock : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>
{
    public DecoderBlock(int features, MultiHeadAttentionBlock selfAttentionBlock, MultiHeadAttentionBlock crossAttentionBlock,
        FeedForwardBlock feedForwardBlock, double dropout) : base(nameof(DecoderBlock))
    {
        _selfAttentionBlock = selfAttentionBlock;
        _crossAttentionBlock = crossAttentionBlock;
        _feedForwardBlock = feedForwardBlock;
        _residualConnection = new ModuleList<ResidualConnection>(new ResidualConnection[]
        {
            new ResidualConnection(features, torch.nn.Dropout(dropout)),
            new ResidualConnection(features, torch.nn.Dropout(dropout)),
            new ResidualConnection(features, torch.nn.Dropout(dropout))
        });

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input, torch.Tensor encoderOutput, torch.Tensor sourceMask,
        torch.Tensor targetMask)
    {
        Func<torch.Tensor, torch.Tensor> subLayer = x => _selfAttentionBlock.forward(x, x, x, targetMask);
        Func<torch.Tensor, torch.Tensor> subLayer2 = x => _crossAttentionBlock.forward(x, encoderOutput, encoderOutput, sourceMask);

        input = _residualConnection[0].forward(input, subLayer);
        input = _residualConnection[1].forward(input, subLayer2);
        input = _residualConnection[2].forward(input, _feedForwardBlock.forward);
        return input;
    }

    private MultiHeadAttentionBlock _selfAttentionBlock;
    private MultiHeadAttentionBlock _crossAttentionBlock;
    private FeedForwardBlock _feedForwardBlock;
    private ModuleList<ResidualConnection> _residualConnection;
}