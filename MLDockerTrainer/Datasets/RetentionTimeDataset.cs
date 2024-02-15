using TorchSharp;

namespace MLDockerTrainer.Datasets
{
    internal class RetentionTimeDataset : torch.utils.data.Dataset
    {
        public override long Count => _integerDataset.Count;


        private int[] _positionIndicativeIndexes = new int[6]
        {
                _padidngIndex, _retentionTimeStartIndex,
                _retentionTimeEndIndex, _sequenceStartIndex,
                _sequenceEndIndex, _maskingIndex
        };

        private int[] _integerIndexes = new[]
        {
            _zeroIndex, _oneIndex, _twoIndex, _threeIndex, _fourIndex, _fiveIndex, _sixIndex, _sevenIndex, _eightIndex,
            _nineIndex, _dot
        };
        public override Dictionary<string, torch.Tensor> GetTensor(long index)
        {
            var data = _integerDataset[(int)index];

            //Encoder Input
            var encoderInputList = data.Select(x => _integerIndexes.Contains(x) ? _maskingIndex : x);
            var encoderInput = torch.tensor(encoderInputList.ToArray());
            //Encoder Input Mask
            var encoderInputMask = encoderInput.tile(new long[] { 100, 1 }).tril(1);
            //Decoder Input
            var decoderInputList = data.Select(x => _integerIndexes.Contains(x) ? x : _maskingIndex);
            var decoderInput = torch.from_array(decoderInputList.ToArray());

            //Decoder Input Mask
            var decoderInputMask = decoderInput.tile(new long[] { 100, 1 }).tril(1);


            //Debug write line tensor to check them
            //Debug.WriteLine(encoderInput.ToString(TensorStringStyle.Julia));
            //Debug.WriteLine(encoderInputMask.ToString(TensorStringStyle.Julia));
            //Debug.WriteLine(decoderInput.ToString(TensorStringStyle.Julia));
            //Debug.WriteLine(decoderInputMask.ToString(TensorStringStyle.Julia));
            //Debug.WriteLine(torch.from_array(data.ToArray()).ToString(TensorStringStyle.Julia));

            return new Dictionary<string, torch.Tensor>()
            {
                { "EncoderInput", encoderInput },
                { "EncoderInputMask", encoderInputMask },
                { "DecoderInput", decoderInput },
                { "DecoderInputMask", decoderInputMask },
                {"Label", torch.from_array(data.Take(22).ToArray())}
            };
        }
        public RetentionTimeDataset(List<List<int>> dataset) : base()
        {
            _integerDataset = dataset;
        }

        private List<List<int>>? _integerDataset;
        private const int _padidngIndex = 0;
        private const int _retentionTimeStartIndex = 1;
        private const int _retentionTimeEndIndex = 2;
        private const int _sequenceStartIndex = 3;
        private const int _sequenceEndIndex = 4;
        private const int _maskingIndex = 5;
        private const int _zeroIndex = 6;
        private const int _oneIndex = 7;
        private const int _twoIndex = 8;
        private const int _threeIndex = 9;
        private const int _fourIndex = 10;
        private const int _fiveIndex = 11;
        private const int _sixIndex = 12;
        private const int _sevenIndex = 13;
        private const int _eightIndex = 14;
        private const int _nineIndex = 15;
        private const int _dot = 16;
    }
}
