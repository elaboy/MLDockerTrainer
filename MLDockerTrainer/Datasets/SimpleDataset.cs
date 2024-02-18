using System.Diagnostics;
using TorchSharp;

namespace MLDockerTrainer.Datasets
{
    public class SimpleDataset : torch.utils.data.Dataset
    {
        public override long Count => _integerDataset.Count;
        public override Dictionary<string, torch.Tensor> GetTensor(long index)
        {
            var data = _integerDataset[(int)index];

            var sequenceTensor = torch.from_array(data.Item1.ToArray(), torch.ScalarType.Float32);

            var label = torch.from_array(new float[] { (float)data.Item2 });

            Debug.WriteLine(sequenceTensor.ToString(TensorStringStyle.Julia));
            Debug.WriteLine(label.ToString(TensorStringStyle.Julia));

            return new Dictionary<string, torch.Tensor>()
            {
                { "Sequence", sequenceTensor },
                {"Target", label}
            };
        }
        public SimpleDataset(List<(List<int>, float)> dataset) : base()
        {
            _integerDataset = dataset;
        }

        private List<(List<int>, float)>? _integerDataset;
    }
}
