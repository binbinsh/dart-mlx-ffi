import 'dart:io';

Future<void> main(List<String> args) async {
  final result = await Process.run('uv', <String>[
    'run',
    'python',
    'benchmark/run_all.py',
    ...args,
  ], workingDirectory: Directory.current.path);

  if (result.stdout case final String out when out.isNotEmpty) {
    stdout.write(out);
  } else if (result.stdout != null) {
    stdout.write(result.stdout);
  }

  if (result.stderr case final String err when err.isNotEmpty) {
    stderr.write(err);
  } else if (result.stderr != null) {
    stderr.write(result.stderr);
  }

  exitCode = result.exitCode;
}
