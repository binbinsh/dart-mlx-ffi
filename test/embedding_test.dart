library;

import 'dart:math' as math;

import 'package:test/test.dart';

import 'package:dart_inference/models.dart';

void main() {
  group('VectorStore', () {
    test('add and retrieve by id', () {
      final store = VectorStore();
      store.add(
        VectorEntry(id: 'doc1', embedding: [1.0, 0.0, 0.0], text: 'hello'),
      );
      expect(store.length, 1);
      expect(store['doc1'], isNotNull);
      expect(store['doc1']!.text, 'hello');
    });

    test('replaces entry with same id', () {
      final store = VectorStore();
      store.add(VectorEntry(id: 'a', embedding: [1, 0, 0], text: 'v1'));
      store.add(VectorEntry(id: 'a', embedding: [0, 1, 0], text: 'v2'));
      expect(store.length, 1);
      expect(store['a']!.text, 'v2');
    });

    test('remove deletes entry', () {
      final store = VectorStore();
      store.add(VectorEntry(id: 'a', embedding: [1, 0, 0]));
      store.remove('a');
      expect(store.length, 0);
      expect(store['a'], isNull);
    });

    test('clear empties store', () {
      final store = VectorStore();
      store.addAll([
        VectorEntry(id: 'a', embedding: [1, 0, 0]),
        VectorEntry(id: 'b', embedding: [0, 1, 0]),
      ]);
      expect(store.length, 2);
      store.clear();
      expect(store.length, 0);
    });

    test('search returns nearest by cosine similarity', () {
      final store = VectorStore();
      // Three orthogonal-ish vectors.
      store.add(VectorEntry(id: 'x', embedding: [1.0, 0.0, 0.0], text: 'x'));
      store.add(VectorEntry(id: 'y', embedding: [0.0, 1.0, 0.0], text: 'y'));
      store.add(VectorEntry(id: 'z', embedding: [0.0, 0.0, 1.0], text: 'z'));

      // Query close to x.
      final results = store.search([0.9, 0.1, 0.0], topK: 2);
      expect(results.length, 2);
      expect(results.first.entry.id, 'x');
      expect(results.first.score, greaterThan(0.9));
    });

    test('search respects minScore', () {
      final store = VectorStore();
      store.add(VectorEntry(id: 'a', embedding: [1.0, 0.0]));
      store.add(VectorEntry(id: 'b', embedding: [0.0, 1.0]));

      // Query [1,0] — 'a' has score 1.0, 'b' has score 0.0.
      final results = store.search([1.0, 0.0], topK: 10, minScore: 0.5);
      expect(results.length, 1);
      expect(results.first.entry.id, 'a');
    });

    test('search respects topK', () {
      final store = VectorStore();
      for (var i = 0; i < 10; i++) {
        final v = List<double>.generate(3, (j) => j == i % 3 ? 1.0 : 0.0);
        store.add(VectorEntry(id: 'doc_$i', embedding: v));
      }
      final results = store.search([1, 0, 0], topK: 3);
      expect(results.length, 3);
    });

    test('addAll builds native index in one batch', () {
      final store = VectorStore();
      store.addAll([
        const VectorEntry(id: 'a', embedding: [1.0, 0.0], text: 'First'),
        const VectorEntry(id: 'b', embedding: [0.8, 0.2], text: 'Second'),
        const VectorEntry(id: 'c', embedding: [0.0, 1.0], text: 'Third'),
      ]);

      expect(store.length, 3);
      final results = store.search([1.0, 0.0], topK: 2);
      expect(results.map((result) => result.entry.id), ['a', 'b']);
    });

    test('addAll validates dimensions before native insert', () {
      final store = VectorStore();
      expect(
        () => store.addAll([
          const VectorEntry(id: 'a', embedding: [1.0, 0.0]),
          const VectorEntry(id: 'b', embedding: [1.0]),
        ]),
        throwsArgumentError,
      );
      expect(store.length, 0);
    });

    test('ragContext builds context string', () {
      final store = VectorStore();
      store.add(
        VectorEntry(id: 'a', embedding: [1.0, 0.0], text: 'First doc.'),
      );
      store.add(
        VectorEntry(id: 'b', embedding: [0.9, 0.1], text: 'Second doc.'),
      );
      store.add(
        VectorEntry(id: 'c', embedding: [0.0, 1.0], text: 'Unrelated.'),
      );

      final ctx = store.ragContext([1.0, 0.0], topK: 2, separator: ' | ');
      expect(ctx, contains('First doc.'));
      expect(ctx, contains('Second doc.'));
      expect(ctx, isNot(contains('Unrelated.')));
    });
  });

  group('VectorEntry', () {
    test('stores all fields', () {
      final entry = VectorEntry(
        id: 'e1',
        embedding: [0.5, 0.5],
        text: 'hello',
        metadata: {'key': 'val'},
      );
      expect(entry.id, 'e1');
      expect(entry.embedding, [0.5, 0.5]);
      expect(entry.text, 'hello');
      expect(entry.metadata, {'key': 'val'});
    });

    test('defaults', () {
      final entry = VectorEntry(id: 'x', embedding: [1]);
      expect(entry.text, '');
      expect(entry.metadata, isEmpty);
    });
  });

  group('SearchResult', () {
    test('stores entry and score', () {
      final entry = VectorEntry(id: 'x', embedding: [1]);
      final result = SearchResult(entry: entry, score: 0.95);
      expect(result.entry.id, 'x');
      expect(result.score, 0.95);
    });
  });

  group('l2Normalise', () {
    test('normalises a vector to unit length', () {
      final vec = [3.0, 4.0];
      l2Normalise(vec);
      final norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
      expect(norm, closeTo(1.0, 1e-10));
      expect(vec[0], closeTo(0.6, 1e-10));
      expect(vec[1], closeTo(0.8, 1e-10));
    });

    test('handles zero vector gracefully', () {
      final vec = [0.0, 0.0, 0.0];
      l2Normalise(vec);
      expect(vec, [0.0, 0.0, 0.0]);
    });

    test('returns the same list', () {
      final vec = [1.0, 0.0];
      expect(identical(l2Normalise(vec), vec), isTrue);
    });
  });

  group('cosine similarity (via search)', () {
    test('identical vectors have similarity 1.0', () {
      final store = VectorStore();
      store.add(VectorEntry(id: 'a', embedding: [1.0, 2.0, 3.0]));
      final results = store.search([1.0, 2.0, 3.0], topK: 1);
      expect(results.first.score, closeTo(1.0, 1e-10));
    });

    test('orthogonal vectors have similarity 0.0', () {
      final store = VectorStore();
      store.add(VectorEntry(id: 'a', embedding: [1.0, 0.0]));
      final results = store.search([0.0, 1.0], topK: 1);
      expect(results.first.score, closeTo(0.0, 1e-10));
    });

    test('opposite vectors have similarity -1.0', () {
      final store = VectorStore();
      store.add(VectorEntry(id: 'a', embedding: [1.0, 0.0]));
      final results = store.search([-1.0, 0.0], topK: 1);
      expect(results.first.score, closeTo(-1.0, 1e-10));
    });

    test('zero-dimensional vectors have similarity 0.0', () {
      final store = VectorStore();
      store.add(const VectorEntry(id: 'empty', embedding: []));
      final results = store.search(const [], topK: 1);
      expect(results.first.entry.id, 'empty');
      expect(results.first.score, 0.0);
    });
  });
}
