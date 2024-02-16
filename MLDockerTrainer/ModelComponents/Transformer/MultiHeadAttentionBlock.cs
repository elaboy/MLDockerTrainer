using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;

namespace MLDockerTrainer.ModelComponents.Transformer;

public class MultiHeadAttentionBlock : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>
{
    public MultiHeadAttentionBlock(int dModel, int numHeads, torch.Tensor mask = null, double dropout = 0.1) : base(nameof(MultiHeadAttentionBlock))
    {
        _dModel = dModel;
        _numHeads = numHeads;
        _dropout = torch.nn.Dropout(dropout);

        Debug.Assert(dModel % numHeads == 0);

        _dK = dModel / numHeads;

        _linearQ = torch.nn.Linear(dModel, dModel, false); //Wq
        torch.nn.init.xavier_uniform_(_linearQ.weight);
        _linearK = torch.nn.Linear(dModel, dModel, false); //Wk
        torch.nn.init.xavier_uniform_(_linearK.weight);
        _linearV = torch.nn.Linear(dModel, dModel, false); //Wv
        _linearO = torch.nn.Linear(dModel, dModel, false); //Wo

        _mask = mask;

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor query, torch.Tensor key, torch.Tensor value, torch.Tensor mask)
    {
        // (BatchSize, SequenceLength, dModel) to (BatchSize, SequenceLength, dModel)
        // to (BatchSize, SequenceLength, numHeads, dK) to (BatchSize, numHeads, SequenceLength, dK)
        var q = _linearQ.forward(query).view(query.shape[0], query.shape[1], _numHeads, _dK).transpose(1, 2);
        var k = _linearK.forward(key).view(key.shape[0], key.shape[1], _numHeads, _dK).transpose(1, 2);
        var v = _linearV.forward(value).view(value.shape[0], value.shape[1], _numHeads, _dK).transpose(1, 2);

        var (x, attention) = Attention(q, k, v, _mask, _dropout);

        //(BatchSize, numHeads, SequenceLength, dK) to (BatchSize, SequenceLength, numHeads, dK) to (BatchSize, SequenceLength, dModel)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], _numHeads * _dK);

        //(Batch, SequenceLength, dModel) to (Batch, SequenceLength, dModel)
        return _linearO.forward(x);
    }

    private static (torch.Tensor, torch.Tensor) Attention(torch.Tensor q, torch.Tensor k, torch.Tensor v, torch.Tensor? mask, Dropout dropout = null)
    {
        var dK = q.shape[2];
        //(BatchSize, numHeads, SequenceLength, dK) to (BatchSize, numHeads, SequenceLength, SequenceLength)
        var scores = torch.matmul(q, k.transpose(-2, -1)) / Math.Sqrt(dK);

        if (mask.shape != null)
            scores.masked_fill(mask == 0, -1e9);

        //(Batch, numHeads, SequenceLength, SequenceLength)
        var attention = torch.nn.functional.softmax(scores, -1);

        if (dropout != null)
            attention = dropout.forward(attention);

        return (torch.matmul(attention, v), scores);
    }

    private int _dModel;
    private int _numHeads;
    private int _dK;
    private Dropout _dropout;
    private Linear _linearQ;
    private Linear _linearK;
    private Linear _linearV;
    private Linear _linearO;
    private torch.Tensor? _mask;
}