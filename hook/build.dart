import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:logging/logging.dart';

void main(List<String> arguments) async {
  Logger.root.level = Level.ALL;
  Logger.root.onRecord.listen(
    // ignore: avoid_print
    (record) => print(record.message),
  );
  final logger = Logger('dart_mlx_ffi');

  await build(arguments, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final code = input.config.code;
    if (code.linkModePreference == LinkModePreference.static) {
      throw UnsupportedError(
        'dart_mlx_ffi requires bundled dynamic libraries for its MLX backend.',
      );
    }

    final packageRoot = input.packageRoot;
    final packageRootPath = packageRoot.toFilePath();

    if (!_supportsMlx(code.targetOS)) {
      throw UnsupportedError('dart_mlx_ffi only supports iOS and macOS.');
    }

    await _buildMlxAsset(
      logger,
      input: input,
      output: output,
      packageRoot: packageRoot,
      packageRootPath: packageRootPath,
      code: code,
    );

    output.dependencies.addAll(await _collectDependencies(packageRoot));
  });
}

Future<void> _buildMlxAsset(
  Logger logger, {
  required BuildInput input,
  required BuildOutputBuilder output,
  required Uri packageRoot,
  required String packageRootPath,
  required CodeConfig code,
}) async {
  final outputDirectory = input.outputDirectory;
  final outputDirectoryPath = outputDirectory.toFilePath();
  final libraryName = code.targetOS.libraryFileName(
    input.packageName,
    DynamicLoadingBundled(),
  );
  final libraryFile = outputDirectory.resolve(libraryName);
  final sdkName = code.targetOS == OS.iOS
      ? _iosSdkName(code.iOS.targetSdk)
      : 'macosx';
  final metalEnabled = await _resolveMetalSupport(logger, code, sdkName);
  // Private ANE bridge requires ARM (NEON intrinsics, Apple Neural Engine).
  // Force it off when targeting x64 regardless of env flag.
  final privateAneEnabled = code.targetArchitecture == Architecture.x64
      ? false
      : _envFlag('DART_MLX_ENABLE_PRIVATE_ANE', defaultValue: true);
  final buildDirectory = outputDirectory.resolve(
    privateAneEnabled
        ? 'cmake_mlx_private_ane_on/'
        : 'cmake_mlx_private_ane_off/',
  );
  final buildDirectoryPath = buildDirectory.toFilePath();
  if (!privateAneEnabled) {
    logger.warning(
      'Private ANE bridge is disabled via DART_MLX_ENABLE_PRIVATE_ANE=0. '
      'Building stub-only private ANE bindings.',
    );
  }

  await Directory.fromUri(buildDirectory).create(recursive: true);

  final compiler = code.cCompiler?.compiler.toFilePath();
  final archiver = code.cCompiler?.archiver.toFilePath();
  final cxxCompiler = _deriveCppCompiler(compiler);
  final configureArgs = <String>[
    '-S',
    packageRoot.resolve('native').toFilePath(),
    '-B',
    buildDirectoryPath,
    '-G',
    'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$outputDirectoryPath',
    '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$outputDirectoryPath',
    '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$outputDirectoryPath',
    '-DCMAKE_OSX_ARCHITECTURES=${_appleArchitectureName(code.targetArchitecture)}',
    '-DCMAKE_OSX_DEPLOYMENT_TARGET=${_deploymentTarget(code)}',
    '-DMLX_BUILD_METAL=${metalEnabled ? 'ON' : 'OFF'}',
    '-DDART_MLX_ENABLE_PRIVATE_ANE=${privateAneEnabled ? 'ON' : 'OFF'}',
    if (code.targetOS == OS.iOS) ...[
      '-DCMAKE_SYSTEM_NAME=iOS',
      '-DCMAKE_OSX_SYSROOT=$sdkName',
      '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY',
    ] else ...[
      '-DCMAKE_OSX_SYSROOT=$sdkName',
    ],
    if (compiler != null) '-DCMAKE_C_COMPILER=$compiler',
    if (cxxCompiler != null) '-DCMAKE_CXX_COMPILER=$cxxCompiler',
    if (archiver != null) '-DCMAKE_AR=$archiver',
  ];

  await _runProcess(
    logger,
    code: code,
    executable: 'cmake',
    arguments: configureArgs,
    workingDirectory: packageRootPath,
  );
  await _runProcess(
    logger,
    code: code,
    executable: 'cmake',
    arguments: [
      '--build',
      buildDirectoryPath,
      '--config',
      'Release',
      '--parallel',
    ],
    workingDirectory: packageRootPath,
  );

  if (!File.fromUri(libraryFile).existsSync()) {
    throw StateError(
      'Expected MLX native library was not produced: $libraryFile',
    );
  }

  output.assets.code.add(
    CodeAsset(
      package: input.packageName,
      name: '${input.packageName}_bindings_generated.dart',
      linkMode: DynamicLoadingBundled(),
      file: libraryFile,
    ),
  );
}

Future<void> _runProcess(
  Logger logger, {
  required CodeConfig code,
  required String executable,
  required List<String> arguments,
  required String workingDirectory,
}) async {
  logger.info('$executable ${arguments.join(' ')}');

  late final Process process;
  if (code.targetOS == OS.windows &&
      code.cCompiler?.windows.developerCommandPrompt != null) {
    final prompt = code.cCompiler!.windows.developerCommandPrompt!;
    final script = _cmdQuote(prompt.script.toFilePath());
    final promptArgs = prompt.arguments.map(_cmdQuote).join(' ');
    final command = [
      'call',
      script,
      if (promptArgs.isNotEmpty) promptArgs,
      '&&',
      _cmdQuote(executable),
      ...arguments.map(_cmdQuote),
    ].join(' ');
    process = await Process.start('cmd.exe', [
      '/d',
      '/s',
      '/c',
      command,
    ], workingDirectory: workingDirectory);
  } else {
    process = await Process.start(
      executable,
      arguments,
      workingDirectory: workingDirectory,
    );
  }

  final stdoutFuture = process.stdout
      .transform(SystemEncoding().decoder)
      .transform(const LineSplitter())
      .forEach(logger.info);
  final stderrFuture = process.stderr
      .transform(SystemEncoding().decoder)
      .transform(const LineSplitter())
      .forEach(logger.severe);

  final exitCode = await process.exitCode;
  await Future.wait([stdoutFuture, stderrFuture]);
  if (exitCode != 0) {
    throw ProcessException(
      executable,
      arguments,
      'Exit code $exitCode',
      exitCode,
    );
  }
}

Future<bool> _hasMetalToolchain(Logger logger, String sdkName) async {
  final result = await Process.run('xcrun', ['-sdk', sdkName, 'metal', '-v']);
  if (result.exitCode == 0) {
    return true;
  }
  logger.warning(
    'Metal toolchain is unavailable for $sdkName. '
    'Building MLX with MLX_BUILD_METAL=OFF. '
    'Install it with: xcodebuild -downloadComponent MetalToolchain',
  );
  return false;
}

Future<bool> _resolveMetalSupport(
  Logger logger,
  CodeConfig code,
  String sdkName,
) async {
  if (code.targetOS == OS.iOS && code.iOS.targetSdk == IOSSdk.iPhoneSimulator) {
    logger.warning(
      'Metal is disabled for iphonesimulator builds. '
      'The simulator toolchain currently produces incompatible deployment '
      'flags when compiling MLX Metal kernels.',
    );
    return false;
  }
  return _hasMetalToolchain(logger, sdkName);
}

Future<Set<Uri>> _collectDependencies(Uri packageRoot) async {
  final dependencies = <Uri>{};
  for (final relativePath in const [
    'native',
    'third_party',
    'hook/build.dart',
  ]) {
    final uri = packageRoot.resolve(relativePath);
    final type = FileSystemEntity.typeSync(uri.toFilePath());
    if (type == FileSystemEntityType.notFound) {
      continue;
    }
    if (type == FileSystemEntityType.file) {
      dependencies.add(uri);
      continue;
    }
    await for (final entity in Directory.fromUri(uri).list(recursive: true)) {
      if (entity is File) {
        dependencies.add(entity.uri);
      }
    }
  }
  return dependencies;
}

bool _supportsMlx(OS os) => os == OS.iOS || os == OS.macOS;

String _appleArchitectureName(Architecture architecture) =>
    switch (architecture) {
      Architecture.arm64 => 'arm64',
      Architecture.x64 => 'x86_64',
      Architecture.ia32 => throw UnsupportedError(
        'ia32 is unsupported for Apple targets.',
      ),
      Architecture.arm => throw UnsupportedError(
        'arm is unsupported for Apple targets.',
      ),
      Architecture.riscv32 => throw UnsupportedError('riscv32 is unsupported.'),
      Architecture.riscv64 => throw UnsupportedError('riscv64 is unsupported.'),
      Architecture() => throw UnsupportedError(
        'Unsupported architecture: ${architecture.name}',
      ),
    };

String _deploymentTarget(CodeConfig code) {
  if (code.targetOS == OS.iOS) {
    return '${math.max(code.iOS.targetVersion, 17)}.0';
  }
  return '${math.max(code.macOS.targetVersion, 14)}.0';
}

String _iosSdkName(IOSSdk sdk) => switch (sdk) {
  IOSSdk.iPhoneOS => 'iphoneos',
  IOSSdk.iPhoneSimulator => 'iphonesimulator',
  IOSSdk() => throw UnsupportedError('Unsupported iOS SDK: $sdk'),
};

String? _deriveCppCompiler(String? cCompiler) {
  if (cCompiler == null) {
    return null;
  }
  if (cCompiler.endsWith('clang')) {
    final candidate = '$cCompiler++';
    if (File(candidate).existsSync()) {
      return candidate;
    }
  }
  return cCompiler;
}

String _cmdQuote(String value) => '"${value.replaceAll('"', '""')}"';

bool _envFlag(String key, {required bool defaultValue}) {
  final raw = Platform.environment[key];
  if (raw == null || raw.isEmpty) {
    return defaultValue;
  }
  final normalized = raw.trim().toLowerCase();
  if (normalized == '1' ||
      normalized == 'true' ||
      normalized == 'yes' ||
      normalized == 'on') {
    return true;
  }
  if (normalized == '0' ||
      normalized == 'false' ||
      normalized == 'no' ||
      normalized == 'off') {
    return false;
  }
  throw FormatException('Invalid boolean environment value for $key: $raw');
}
