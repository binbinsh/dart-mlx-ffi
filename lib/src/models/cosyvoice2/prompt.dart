import 'package:dart_inference/dart_mlx_ffi.dart';

final class CosyVoice2PromptBundle {
  CosyVoice2PromptBundle._({
    required this.promptSpeechToken,
    required this.promptSpeechTokenLen,
    required this.promptMel,
    required this.promptMelLen,
    required this.speakerEmbedding,
  });

  factory CosyVoice2PromptBundle.load(String bundlePath) {
    final tensors = mx.io
        .loadSafetensors('$bundlePath/prompt.safetensors')
        .tensors;
    final promptSpeechToken = tensors['prompt_speech_token'];
    final promptSpeechTokenLen = tensors['prompt_speech_token_len'];
    final promptMel = tensors['prompt_mel'];
    final promptMelLen = tensors['prompt_mel_len'];
    final speakerEmbedding = tensors['speaker_embedding'];
    if (promptSpeechToken == null ||
        promptSpeechTokenLen == null ||
        promptMel == null ||
        promptMelLen == null ||
        speakerEmbedding == null) {
      throw StateError('Incomplete CosyVoice2 prompt bundle in $bundlePath');
    }
    return CosyVoice2PromptBundle._(
      promptSpeechToken: promptSpeechToken,
      promptSpeechTokenLen: promptSpeechTokenLen,
      promptMel: promptMel,
      promptMelLen: promptMelLen,
      speakerEmbedding: speakerEmbedding,
    );
  }

  final MlxArray promptSpeechToken;
  final MlxArray promptSpeechTokenLen;
  final MlxArray promptMel;
  final MlxArray promptMelLen;
  final MlxArray speakerEmbedding;

  void close() {
    promptSpeechToken.close();
    promptSpeechTokenLen.close();
    promptMel.close();
    promptMelLen.close();
    speakerEmbedding.close();
  }
}
