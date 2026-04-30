import 'dart:convert';
import 'dart:io';

import 'package:dart_inference/models.dart';

void main(List<String> args) {
  if (args.length < 2) {
    stderr.writeln(
      'Usage: dart run tool/sarashina2_tokenizer_encode.dart '
      '<tokenizer.sara2tok> <text> [<text> ...]',
    );
    exitCode = 64;
    return;
  }
  final tokenizer = Sarashina2BaseTokenizer.fromFile(args.first);
  try {
    final rows = [
      for (final text in args.skip(1))
        {'text': text, 'ids': tokenizer.encode(text).toList(growable: false)},
    ];
    stdout.writeln(jsonEncode(rows));
  } finally {
    tokenizer.close();
  }
}
