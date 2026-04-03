part of 'qwen3_5.dart';

typedef Qwen35TopLogit = ({int tokenId, double logit});

extension Qwen35TopK on Qwen3_5Runner {
  List<Qwen35TopLogit> topLogitsForPrefix(List<int> tokenIds, {int topK = 10}) {
    final logits = runFullLogits(tokenIds).astype(MlxDType.MLX_FLOAT32);
    try {
      final values = logits.toList().cast<double>();
      final pairs = <Qwen35TopLogit>[
        for (var index = 0; index < values.length; index++)
          (tokenId: index, logit: values[index]),
      ]..sort((a, b) => b.logit.compareTo(a.logit));
      return pairs.take(topK).toList(growable: false);
    } finally {
      logits.close();
    }
  }
}
