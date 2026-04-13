@TestOn('mac-os')
library;

import 'package:test/test.dart';

import 'package:dart_mlx_ffi/models.dart';

void main() {
  group('TuningKnob', () {
    test('resolves macOS default on macOS', () {
      final knob = intKnob(name: 'test', iosDefault: 100, macosDefault: 200);
      // We're running on macOS in tests.
      expect(knob.resolve(ApplePlatform.macos), 200);
    });

    test('resolves iOS default when asked', () {
      final knob = intKnob(name: 'test', iosDefault: 100, macosDefault: 200);
      expect(knob.resolve(ApplePlatform.ios), 100);
    });

    test('double knob works', () {
      final knob = doubleKnob(name: 'temp', iosDefault: 0.7, macosDefault: 1.0);
      expect(knob.resolve(ApplePlatform.macos), 1.0);
      expect(knob.resolve(ApplePlatform.ios), 0.7);
    });

    test('bool knob works', () {
      final knob = boolKnob(
        name: 'flag',
        iosDefault: true,
        macosDefault: false,
      );
      expect(knob.resolve(ApplePlatform.macos), isFalse);
      expect(knob.resolve(ApplePlatform.ios), isTrue);
    });

    test('string knob works', () {
      final knob = stringKnob(
        name: 'scheme',
        iosDefault: 'uniform',
        macosDefault: 'none',
      );
      expect(knob.resolve(ApplePlatform.macos), 'none');
      expect(knob.resolve(ApplePlatform.ios), 'uniform');
    });
  });

  group('TuningProfile', () {
    test('resolves all knobs for a platform', () {
      final profile = TuningProfile(
        name: 'test',
        knobs: {
          'chunk': intKnob(name: 'chunk', iosDefault: 256, macosDefault: 1024),
          'fast': boolKnob(name: 'fast', iosDefault: true, macosDefault: false),
        },
      );

      final macos = profile.resolve(ApplePlatform.macos);
      expect(macos['chunk'], 1024);
      expect(macos['fast'], false);

      final ios = profile.resolve(ApplePlatform.ios);
      expect(ios['chunk'], 256);
      expect(ios['fast'], true);
    });

    test('get<T> returns typed value', () {
      final profile = TuningProfile(
        name: 'test',
        knobs: {'x': intKnob(name: 'x', iosDefault: 1, macosDefault: 2)},
      );
      expect(profile.get<int>('x', ApplePlatform.macos), 2);
    });

    test('get throws for unknown knob', () {
      final profile = TuningProfile(name: 'test', knobs: {});
      expect(() => profile.get<int>('missing'), throwsArgumentError);
    });
  });

  group('RuntimeTuning', () {
    test('register and resolve', () {
      final tuning = RuntimeTuning.instance;
      final profile = TuningProfile(
        name: 'test_model',
        knobs: {'val': intKnob(name: 'val', iosDefault: 10, macosDefault: 20)},
      );

      tuning.register('__test__', profile);
      expect(tuning['__test__'], isNotNull);
      expect(tuning.resolve<int>('__test__', 'val', ApplePlatform.macos), 20);
    });

    test('resolve throws for unregistered model', () {
      expect(
        () => RuntimeTuning.instance.resolve<int>('__no_such_model__', 'val'),
        throwsStateError,
      );
    });

    test('dumpAll returns all resolved values', () {
      final tuning = RuntimeTuning.instance;
      tuning.register(
        '__dump_test__',
        TuningProfile(
          name: 'dump',
          knobs: {'a': intKnob(name: 'a', iosDefault: 1, macosDefault: 2)},
        ),
      );
      final dump = tuning.dumpAll(ApplePlatform.macos);
      expect(dump['__dump_test__']!['a'], 2);
    });
  });

  group('built-in profiles', () {
    test('paddleOcrVlTuning has expected knobs', () {
      expect(paddleOcrVlTuning.knobs.containsKey('max_pixels'), isTrue);
      expect(paddleOcrVlTuning.knobs.containsKey('prefill_chunk_size'), isTrue);
      expect(
        paddleOcrVlTuning.knobs.containsKey('kv_cache_quant_bits'),
        isTrue,
      );
      expect(
        paddleOcrVlTuning.get<int>('max_pixels', ApplePlatform.ios),
        512 * 512,
      );
    });

    test('qwen35Tuning has expected knobs', () {
      expect(qwen35Tuning.knobs.containsKey('prefill_chunk_size'), isTrue);
      expect(qwen35Tuning.knobs.containsKey('max_kv_seq_len'), isTrue);
      expect(
        qwen35Tuning.get<int>('max_kv_seq_len', ApplePlatform.macos),
        8192,
      );
    });

    test('qwen3AsrTuning has expected knobs', () {
      expect(qwen3AsrTuning.knobs.containsKey('max_audio_tokens'), isTrue);
      expect(qwen3AsrTuning.knobs.containsKey('max_new_tokens'), isTrue);
    });
  });

  group('currentPlatform', () {
    test('returns macos on macOS test runner', () {
      // Tests run on macOS.
      expect(currentPlatform, ApplePlatform.macos);
    });
  });
}
