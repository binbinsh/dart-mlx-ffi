import 'package:test/test.dart';

import 'package:dart_inference/models.dart';

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
}
