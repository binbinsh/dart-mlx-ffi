import 'dart:ffi' as ffi;

import 'package:test/test.dart';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';

void main() {
  test('strips SSML sub aliases for TTS text', () {
    final ssml =
        '<speak>I paid <sub alias="one hundred dollars">\$100</sub>.</speak>';

    expect(stripSsmlForTts(ssml), 'I paid one hundred dollars.');
  });

  test('applies simple English and Chinese TN fallbacks', () {
    expect(verbalizeEnglish('\$123'), 'one hundred twenty three dollars');
    expect(verbalizeChinese('2026'), '二零二六');
  });

  test('detects Chinese text for phonemizer language selection', () {
    expect(looksChinese('hello'), isFalse);
    expect(looksChinese('你好'), isTrue);
  });

  test('proposes pronunciation targets and candidate ids', () {
    final resolver = PronunciationTargetResolver(
      homographPronunciations: const ['lead_noun', 'lead_verb'],
      polyphonePronunciations: const ['xing2', 'hang2'],
      homographSurfaceCandidates: const {
        'Lead': ['lead_noun', 'lead_verb'],
      },
      polyphoneSurfaceCandidates: const {
        '行': ['xing2', 'hang2'],
      },
    );

    final homographs = resolver.proposeHomographs('Lead the team.');
    final polyphones = resolver.proposePolyphones('银行行长');

    expect(homographs, hasLength(1));
    expect(homographs.single.surface, 'Lead');
    expect(resolver.homographCandidateIds(homographs.single), [0, 1]);
    expect(polyphones, hasLength(2));
    expect(polyphones.map((item) => item.surface), ['行', '行']);
    expect(resolver.polyphoneCandidateIds(polyphones.first), [1, 0]);
  });

  test('renders phoneme tags in composed SSML', () {
    final ir = FrontendIr()
      ..homographItems.add(
        const PronunciationItem(
          start: 0,
          end: 4,
          surface: 'Lead',
          pronunciation: 'lead_verb',
        ),
      );

    expect(
      composeSsml('Lead us.', ir),
      '<speak><phoneme ph="lead_verb">Lead</phoneme> us.</speak>',
    );
  });

  test('prefers the matching TN language when both heads emit a span', () {
    final english = FrontendIr()
      ..tnEnItems.add(
        TnItem(
          start: 7,
          end: 10,
          surface: '\$12',
          tnType: 'MONEY',
          spoken: 'twelve dollars',
        ),
      )
      ..tnZhItems.add(
        TnItem(
          start: 7,
          end: 10,
          surface: '\$12',
          tnType: 'CARDINAL',
          spoken: '\$一二',
        ),
      );
    expect(
      composeSsml('I paid \$12.', english),
      '<speak>I paid <sub alias="twelve dollars">\$12</sub>.</speak>',
    );

    final chinese = FrontendIr()
      ..tnEnItems.add(
        TnItem(
          start: 3,
          end: 7,
          surface: '2026',
          tnType: 'DATE',
          spoken: 'twenty twenty six',
        ),
      )
      ..tnZhItems.add(
        TnItem(
          start: 3,
          end: 7,
          surface: '2026',
          tnType: 'DIGIT',
          spoken: '二零二六',
        ),
      );
    expect(
      composeSsml('今天是2026年。', chinese),
      '<speak>今天是<sub alias="二零二六">2026</sub>年。</speak>',
    );
  });

  test('composes SSML through native with Dart UTF-16 offsets', () {
    final ir = FrontendIr()
      ..tnZhItems.add(
        TnItem(
          start: 3,
          end: 7,
          surface: '2026',
          tnType: 'DIGIT',
          spoken: '二零二六',
        ),
      );

    expect(
      composeSsml('🙂年2026', ir),
      '<speak>🙂年<sub alias="二零二六">2026</sub></speak>',
    );
  });

  test('decodes native-backed structured logits through Dart helpers', () {
    final buffers = <NativeTensorBuffer>[];
    NativeTensorBuffer tensor(List<int> shape, List<double> values) {
      final buffer = NativeTensorBuffer.float32(shape);
      buffer.asFloat32List().setAll(0, values);
      buffers.add(buffer);
      return buffer;
    }

    List<double> logits(List<int> ids, int classes) {
      final out = List<double>.filled(ids.length * classes, -5);
      for (var i = 0; i < ids.length; i++) {
        out[i * classes + ids[i]] = 5;
      }
      return out;
    }

    try {
      final decoder = StructuredDecoder(
        emotionLabels: const ['sad', 'happy'],
        tnEnTypes: const ['UNKNOWN', 'CARDINAL'],
        tnZhTypes: const [],
        targetResolver: PronunciationTargetResolver(
          homographPronunciations: const ['lead_noun', 'lead_verb'],
          polyphonePronunciations: const [],
          homographSurfaceCandidates: const {},
          polyphoneSurfaceCandidates: const {},
        ),
        englishTnLexicon: const {},
        emphasisThreshold: 0.5,
      );
      final outputs = <String, Object?>{
        'emotion_logits': tensor(const [1, 2], const [-2, 2]).tensor,
        'emphasis_char_logits': tensor(const [
          1,
          6,
          5,
        ], logits(const [4, 0, 0, 0, 0, 0], 5)).tensor,
        'homograph_pron_logits_multi': tensor(
          const [1, 1, 2],
          const [0.1, 0.9],
        ).tensor,
        'tn_en_char_span_logits': tensor(const [
          1,
          6,
          5,
        ], logits(const [0, 0, 0, 0, 0, 4], 5)).tensor,
        'tn_en_char_type_logits': tensor(const [
          1,
          6,
          2,
        ], logits(const [0, 0, 0, 0, 0, 1], 2)).tensor,
      };

      final ir = decoder.decode(
        text: 'Lead 1',
        numChars: 6,
        outputs: outputs,
        homographTargets: const [
          PronunciationItem(
            start: 0,
            end: 4,
            surface: 'Lead',
            pronunciation: '',
          ),
        ],
        polyphoneTargets: const [],
      );

      expect(ir.emotionLabels, ['happy']);
      expect(ir.emphasisSpans.single.toJson(), {
        'start': 0,
        'end': 1,
        'label': 'EMPHASIS',
      });
      expect(ir.homographItems.single.pronunciation, 'lead_verb');
      expect(ir.tnEnItems.single.toJson(), {
        'start': 5,
        'end': 6,
        'surface': '1',
        'tnType': 'CARDINAL',
        'spoken': 'one',
      });
    } finally {
      for (final buffer in buffers) {
        buffer.close();
      }
    }
  });

  test('encodes structured inputs into native-backed tensors', () {
    final builder = StructuredInputBuilder(
      tokenizer: MmBertBpeTokenizer(
        vocab: const {
          '<pad>': 0,
          '<eos>': 1,
          '<bos>': 2,
          '<unk>': 3,
          '▁': 4,
          'L': 5,
          'e': 6,
          'a': 7,
          'd': 8,
        },
        mergeRanks: const {},
        bosId: 2,
        eosId: 1,
        padId: 0,
        unkId: 3,
      ),
      charVocab: CharVocab(
        const {'<pad>': 0, '<unk>': 1, 'L': 2, 'e': 3, 'a': 4, 'd': 5},
        padId: 0,
        unkId: 1,
      ),
      config: StructuredFrontendConfig(
        batchSize: 1,
        tokenLength: 8,
        charLength: 8,
        homographTargets: 1,
        polyphoneTargets: 1,
        numHomographClasses: 2,
        numPolyphoneClasses: 2,
        emphasisThreshold: 0.5,
      ),
      targetResolver: PronunciationTargetResolver(
        homographPronunciations: const ['lead_noun', 'lead_verb'],
        polyphonePronunciations: const [],
        homographSurfaceCandidates: const {
          'Lead': ['lead_noun', 'lead_verb'],
        },
        polyphoneSurfaceCandidates: const {},
      ),
    );
    final encoded = builder.encodeBatch(const ['Lead']);
    try {
      final inputs = encoded.toModelInputs(
        batchSize: 1,
        tokenLength: 8,
        charLength: 8,
        homographTargets: 1,
        polyphoneTargets: 1,
        numHomographClasses: 2,
        numPolyphoneClasses: 2,
      );
      final inputIds = inputs['input_ids']! as RuntimeTensor;
      final homographMasks =
          inputs['homograph_candidate_masks']! as RuntimeTensor;

      expect(inputIds.nativeData, isNot(ffi.nullptr));
      expect(homographMasks.nativeData, isNot(ffi.nullptr));
      expect(inputIds.asInt64List().first, 2);
      expect(encoded.homographTargets.single.single.surface, 'Lead');
    } finally {
      encoded.close();
    }
  });
}
