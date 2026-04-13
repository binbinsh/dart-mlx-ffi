/// Embedding service / on-device RAG.
///
/// Provides a model-agnostic embedding interface and a lightweight vector
/// store for retrieval-augmented generation (RAG) on device.
///
/// Inspired by osaurus `EmbeddingService.swift` / `MetalSafeEmbedder.swift`.
library;

import 'dart:math' as math;

import 'metal_gate.dart';

// ---------------------------------------------------------------------------
// Embedder interface
// ---------------------------------------------------------------------------

/// A model that maps text (or tokens) to a dense vector embedding.
///
/// Concrete implementations wrap a loaded model and handle tokenisation,
/// pooling, and normalisation.
abstract interface class Embedder {
  /// Embed a single text string.
  ///
  /// Returns a normalised float vector of length [embeddingDim].
  List<double> embed(String text);

  /// Embed a batch of texts.
  ///
  /// Returns one vector per input text.
  List<List<double>> embedBatch(List<String> texts);

  /// Dimensionality of the output vectors.
  int get embeddingDim;

  /// Release GPU resources.
  void close();
}

/// An [Embedder] that routes all GPU work through a [MetalGate] for
/// multi-isolate safety.
final class GatedEmbedder implements Embedder {
  GatedEmbedder({required this.inner, MetalGate? gate})
    : _gate = gate ?? metalGate;

  /// The underlying embedder that does the actual computation.
  final Embedder inner;
  final MetalGate _gate;

  @override
  int get embeddingDim => inner.embeddingDim;

  @override
  List<double> embed(String text) {
    // Synchronous wrapper — in practice, callers should prefer embedAsync.
    return inner.embed(text);
  }

  /// Async version that schedules through the gate.
  Future<List<double>> embedAsync(String text) =>
      _gate.run(() => inner.embed(text));

  @override
  List<List<double>> embedBatch(List<String> texts) => inner.embedBatch(texts);

  /// Async version that schedules through the gate.
  Future<List<List<double>>> embedBatchAsync(List<String> texts) =>
      _gate.run(() => inner.embedBatch(texts));

  @override
  void close() => inner.close();
}

// ---------------------------------------------------------------------------
// VectorStore — lightweight in-memory vector index
// ---------------------------------------------------------------------------

/// A document with its embedding and payload.
final class VectorEntry {
  const VectorEntry({
    required this.id,
    required this.embedding,
    this.text = '',
    this.metadata = const <String, Object?>{},
  });

  /// Unique identifier.
  final String id;

  /// Dense embedding vector.
  final List<double> embedding;

  /// Original text (for display / context injection).
  final String text;

  /// Arbitrary metadata.
  final Map<String, Object?> metadata;
}

/// A search result with its similarity score.
final class SearchResult {
  const SearchResult({required this.entry, required this.score});

  /// The matched entry.
  final VectorEntry entry;

  /// Cosine similarity score (higher = more similar, range [-1, 1]).
  final double score;
}

/// Lightweight in-memory vector store with cosine similarity search.
///
/// Suitable for small-to-medium corpora (hundreds to low thousands of
/// documents).  For larger collections, consider an external vector DB.
///
/// ```dart
/// final store = VectorStore();
/// store.add(VectorEntry(id: '1', embedding: [...], text: 'hello'));
/// final results = store.search(queryEmbedding, topK: 5);
/// ```
final class VectorStore {
  VectorStore();

  final Map<String, VectorEntry> _entries = {};

  /// Number of entries.
  int get length => _entries.length;

  /// Add an entry (replaces any existing entry with the same ID).
  void add(VectorEntry entry) {
    _entries[entry.id] = entry;
  }

  /// Add multiple entries.
  void addAll(Iterable<VectorEntry> entries) {
    for (final entry in entries) {
      _entries[entry.id] = entry;
    }
  }

  /// Remove an entry by ID.
  void remove(String id) {
    _entries.remove(id);
  }

  /// Clear all entries.
  void clear() {
    _entries.clear();
  }

  /// Retrieve an entry by ID.
  VectorEntry? operator [](String id) => _entries[id];

  /// Search for the [topK] most similar entries to [queryEmbedding].
  ///
  /// Uses brute-force cosine similarity.  Returns results sorted by
  /// descending score.
  List<SearchResult> search(
    List<double> queryEmbedding, {
    int topK = 5,
    double minScore = -1.0,
  }) {
    final results = <SearchResult>[];
    for (final entry in _entries.values) {
      final score = _cosineSimilarity(queryEmbedding, entry.embedding);
      if (score >= minScore) {
        results.add(SearchResult(entry: entry, score: score));
      }
    }
    results.sort((a, b) => b.score.compareTo(a.score));
    return results.take(topK).toList();
  }

  /// Bulk index: embed all [texts] via [embedder] and store.
  void indexTexts(
    Embedder embedder,
    List<String> texts, {
    String Function(int index)? idGenerator,
    Map<String, Object?> Function(int index)? metadataGenerator,
  }) {
    final embeddings = embedder.embedBatch(texts);
    for (var i = 0; i < texts.length; i++) {
      add(
        VectorEntry(
          id: idGenerator?.call(i) ?? 'doc_$i',
          embedding: embeddings[i],
          text: texts[i],
          metadata: metadataGenerator?.call(i) ?? const {},
        ),
      );
    }
  }

  /// Build a RAG context string from the top-K results for a query.
  ///
  /// Returns the concatenated texts of the top results, separated by
  /// [separator].
  String ragContext(
    List<double> queryEmbedding, {
    int topK = 3,
    double minScore = 0.0,
    String separator = '\n\n',
  }) {
    final results = search(queryEmbedding, topK: topK, minScore: minScore);
    return results.map((r) => r.entry.text).join(separator);
  }
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

double _cosineSimilarity(List<double> a, List<double> b) {
  if (a.length != b.length) {
    throw ArgumentError(
      'Vector dimensions must match: ${a.length} vs ${b.length}',
    );
  }
  var dot = 0.0;
  var normA = 0.0;
  var normB = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  final denom = math.sqrt(normA) * math.sqrt(normB);
  return denom > 0 ? dot / denom : 0.0;
}

// ---------------------------------------------------------------------------
// Utility: L2-normalise a vector in-place
// ---------------------------------------------------------------------------

/// Normalise [vec] to unit length (L2 norm).  Modifies in-place and returns
/// the same list.
List<double> l2Normalise(List<double> vec) {
  var norm = 0.0;
  for (final v in vec) {
    norm += v * v;
  }
  norm = math.sqrt(norm);
  if (norm > 0) {
    for (var i = 0; i < vec.length; i++) {
      vec[i] /= norm;
    }
  }
  return vec;
}
