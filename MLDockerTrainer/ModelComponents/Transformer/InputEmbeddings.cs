using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;

namespace MLDockerTrainer.ModelComponents.Transformer;

public class InputEmbeddings : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    public InputEmbeddings(int dModel, int vocabSize) : base(nameof(InputEmbeddings))
    {
        _embedding = torch.nn.Embedding(vocabSize, dModel, padding_idx: 0);
        _dModel = dModel;
        _vocabSize = vocabSize;

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input)
    {
        //Debug.WriteLine(input.ToString(TensorStringStyle.Julia));
        return _embedding.forward(input) * Math.Sqrt(_dModel);
    }

    private Embedding _embedding;
    private int _dModel;
    private int _vocabSize;
}