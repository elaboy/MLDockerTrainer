using TorchSharp;

namespace MLDockerTrainer.ModelComponents.Transformer
{
    public class Transformer : torch.nn.Module<torch.Tensor, torch.Tensor>
    {
        public Transformer(Encoder encoder, Decoder decoder, InputEmbeddings sourceEmbedding,
            InputEmbeddings targetEmbedding, PositionalEncoder sourcePosition, PositionalEncoder targetPosition,
            ProjectionLayer projectionLayer) : base(nameof(Transformer))
        {
            _encoder = encoder;
            _decoder = decoder;
            _sourceEmbedding = sourceEmbedding;
            _targetEmbedding = targetEmbedding;
            _sourcePosition = sourcePosition;
            _targetPosition = targetPosition;
            _projectionLayer = projectionLayer;
            SourceVocabSize = 2729; //todo: look for a better way to do this instead of a fixed value

            RegisterComponents();
        }

        public torch.Tensor Encode(torch.Tensor source, torch.Tensor sourceMask)
        {
            source = _sourceEmbedding.forward(source);
            source = _sourcePosition.forward(source);
            return _encoder.forward(source, sourceMask);
        }

        public torch.Tensor Decode(torch.Tensor encoderOutput, torch.Tensor sourceMask, torch.Tensor target,
            torch.Tensor targetMask)
        {
            target = _targetEmbedding.forward(target);
            target = _targetPosition.forward(target);
            return _decoder.forward(target, encoderOutput, sourceMask, targetMask);
        }

        public torch.Tensor Project(torch.Tensor decoderOutput)
        {
            return _projectionLayer.forward(decoderOutput);
        }

        public override torch.Tensor forward(torch.Tensor input)
        {
            throw new NotImplementedException();
        }

        private Encoder _encoder;
        private Decoder _decoder;
        private InputEmbeddings _sourceEmbedding;
        private InputEmbeddings _targetEmbedding;
        private PositionalEncoder _sourcePosition;
        private PositionalEncoder _targetPosition;
        private ProjectionLayer _projectionLayer;
        public int SourceVocabSize { get; set; }

    }
}
