/// Embedding service / on-device RAG.
///
/// Provides a model-agnostic embedding interface and a lightweight vector
/// store for retrieval-augmented generation (RAG) on device.
///
/// Inspired by osaurus `EmbeddingService.swift` / `MetalSafeEmbedder.swift`.
library;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../../runtime/native_bindings.dart' as native;
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
  _NativeVectorIndex? _index;
  int? _dimension;

  /// Number of entries.
  int get length => _entries.length;

  /// Add an entry (replaces any existing entry with the same ID).
  void add(VectorEntry entry) {
    _ensureIndex(entry.embedding.length).put(entry.id, entry.embedding);
    _entries[entry.id] = entry;
  }

  /// Add multiple entries.
  void addAll(Iterable<VectorEntry> entries) {
    final items = entries.toList(growable: false);
    if (items.isEmpty) {
      return;
    }
    final dimension = items.first.embedding.length;
    for (final entry in items) {
      if (entry.embedding.length != dimension) {
        throw ArgumentError(
          'Vector dimensions must match: ${entry.embedding.length} vs '
          '$dimension',
        );
      }
    }
    _ensureIndex(dimension).putMany(items, dimension);
    for (final entry in items) {
      _entries[entry.id] = entry;
    }
  }

  /// Remove an entry by ID.
  void remove(String id) {
    _index?.remove(id);
    _entries.remove(id);
  }

  /// Clear all entries.
  void clear() {
    _index?.close();
    _index = null;
    _dimension = null;
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
    if (_entries.isEmpty || topK <= 0) {
      return const [];
    }
    final dimension = _dimension;
    if (dimension == null) {
      return const [];
    }
    if (queryEmbedding.length != dimension) {
      throw ArgumentError(
        'Vector dimensions must match: ${queryEmbedding.length} vs $dimension',
      );
    }
    final hits = _index!.search(queryEmbedding, topK: topK, minScore: minScore);
    return [
      for (final hit in hits)
        if (_entries[hit.id] case final entry?)
          SearchResult(entry: entry, score: hit.score),
    ];
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

  /// Release native vector index memory immediately.
  void close() {
    clear();
  }

  _NativeVectorIndex _ensureIndex(int dimension) {
    final current = _dimension;
    if (current != null) {
      if (dimension != current) {
        throw ArgumentError(
          'Vector dimensions must match: $dimension vs $current',
        );
      }
      return _index!;
    }
    final index = _NativeVectorIndex(dimension);
    _index = index;
    _dimension = dimension;
    return index;
  }
}

final _vectorFinalizer = Finalizer<ffi.Pointer<ffi.Void>>((handle) {
  if (handle != ffi.nullptr) {
    native.vecFree(handle);
  }
});

final class _VectorHit {
  const _VectorHit(this.id, this.score);

  final String id;
  final double score;
}

final class _NativeVectorIndex {
  _NativeVectorIndex(int dimension) : _handle = native.vecNew(dimension) {
    if (_handle == ffi.nullptr) {
      throw StateError('Failed to create native vector index.');
    }
    _vectorFinalizer.attach(this, _handle, detach: this);
  }

  ffi.Pointer<ffi.Void> _handle;

  void put(String id, List<double> values) {
    _checkOpen();
    final idPtr = id.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final valuesPtr = _doubleList(values);
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      final status = native.vecPut(
        _handle,
        idPtr,
        valuesPtr,
        values.length,
        error,
      );
      if (status != 0) {
        throw StateError(_takeVectorError(error));
      }
    } finally {
      calloc
        ..free(idPtr)
        ..free(valuesPtr)
        ..free(error);
    }
  }

  void putMany(List<VectorEntry> entries, int dimension) {
    _checkOpen();
    final ids = calloc<ffi.Pointer<ffi.Char>>(entries.length);
    final strings = <ffi.Pointer<ffi.Char>>[];
    final valueCount = entries.length * dimension;
    final valuesPtr = valueCount == 0
        ? ffi.nullptr
        : calloc<ffi.Double>(valueCount);
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      for (var i = 0; i < entries.length; i += 1) {
        final id = entries[i].id
            .toNativeUtf8(allocator: calloc)
            .cast<ffi.Char>();
        strings.add(id);
        ids[i] = id;
      }
      if (valueCount > 0) {
        final values = valuesPtr.asTypedList(valueCount);
        var offset = 0;
        for (final entry in entries) {
          values.setAll(offset, entry.embedding);
          offset += dimension;
        }
      }
      final status = native.vecPutMany(
        _handle,
        ids,
        valuesPtr,
        entries.length,
        dimension,
        error,
      );
      if (status != 0) {
        throw StateError(_takeVectorError(error));
      }
    } finally {
      for (final string in strings) {
        calloc.free(string);
      }
      calloc.free(ids);
      if (valuesPtr != ffi.nullptr) {
        calloc.free(valuesPtr);
      }
      calloc.free(error);
    }
  }

  void remove(String id) {
    _checkOpen();
    final idPtr = id.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    try {
      native.vecRemove(_handle, idPtr);
    } finally {
      calloc.free(idPtr);
    }
  }

  List<_VectorHit> search(
    List<double> query, {
    required int topK,
    required double minScore,
  }) {
    _checkOpen();
    final queryPtr = _doubleList(query);
    final results = calloc<ffi.Pointer<native.VecResultAbi>>();
    final count = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      final status = native.vecSearch(
        _handle,
        queryPtr,
        query.length,
        topK,
        minScore,
        results,
        count,
        error,
      );
      if (status != 0) {
        throw StateError(_takeVectorError(error));
      }
      final resultPtr = results.value;
      final length = count.value;
      if (resultPtr == ffi.nullptr || length <= 0) {
        return const [];
      }
      return [
        for (var i = 0; i < length; i += 1)
          _VectorHit(
            resultPtr[i].id.cast<Utf8>().toDartString(),
            resultPtr[i].score,
          ),
      ];
    } finally {
      if (results.value != ffi.nullptr) {
        native.vecFreeResults(results.value, count.value);
      }
      calloc
        ..free(queryPtr)
        ..free(results)
        ..free(count)
        ..free(error);
    }
  }

  void close() {
    final handle = _handle;
    if (handle == ffi.nullptr) {
      return;
    }
    _handle = ffi.nullptr;
    _vectorFinalizer.detach(this);
    native.vecFree(handle);
  }

  void _checkOpen() {
    if (_handle == ffi.nullptr) {
      throw StateError('Native vector index is closed.');
    }
  }
}

ffi.Pointer<ffi.Double> _doubleList(List<double> values) {
  final pointer = calloc<ffi.Double>(values.length);
  pointer.asTypedList(values.length).setAll(0, values);
  return pointer;
}

String _takeVectorError(ffi.Pointer<ffi.Pointer<ffi.Char>> error) {
  final value = error.value;
  if (value == ffi.nullptr) return 'Native vector index call failed.';
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    native.freeStr(value);
    error.value = ffi.nullptr;
  }
}

// ---------------------------------------------------------------------------
// Utility: L2-normalise a vector in-place
// ---------------------------------------------------------------------------

/// Normalise [vec] to unit length (L2 norm).  Modifies in-place and returns
/// the same list.
List<double> l2Normalise(List<double> vec) {
  if (vec.isEmpty) {
    return vec;
  }
  final valuesPtr = _doubleList(vec);
  try {
    final status = native.vecL2Norm(valuesPtr, vec.length);
    if (status != 0) {
      throw StateError('Native vector normalization failed.');
    }
    vec.setAll(0, valuesPtr.asTypedList(vec.length));
  } finally {
    calloc.free(valuesPtr);
  }
  return vec;
}
