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
  final logger = Logger('dart_inference');

  await build(arguments, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final code = input.config.code;
    if (code.linkModePreference == LinkModePreference.static) {
      throw UnsupportedError(
        'dart_inference requires bundled dynamic libraries for its Zig runtime.',
      );
    }

    final packageRoot = input.packageRoot;
    final packageRootPath = packageRoot.toFilePath();

    if (!_supportsRuntime(code.targetOS)) {
      throw UnsupportedError(
        'dart_inference runtime supports iOS, macOS, Windows, Linux, and Android.',
      );
    }

    await _buildRuntimeAsset(
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

Future<void> _buildRuntimeAsset(
  Logger logger, {
  required BuildInput input,
  required BuildOutputBuilder output,
  required Uri packageRoot,
  required String packageRootPath,
  required CodeConfig code,
}) async {
  final outputDirectory = input.outputDirectory;
  final outputDirectoryPath = outputDirectory.toFilePath();
  final runtimeLibraryName = code.targetOS.libraryFileName(
    '${input.packageName}_runtime',
    DynamicLoadingBundled(),
  );
  final runtimeLibraryFile = outputDirectory.resolve(runtimeLibraryName);
  final adapterLibraryName = code.targetOS.libraryFileName(
    '${input.packageName}_runtime_adapter',
    DynamicLoadingBundled(),
  );
  final adapterLibraryFile = outputDirectory.resolve(adapterLibraryName);
  final buildDirectory = outputDirectory.resolve('cmake_runtime/');
  final buildDirectoryPath = buildDirectory.toFilePath();

  await Directory.fromUri(buildDirectory).create(recursive: true);

  final compiler = code.cCompiler?.compiler.toFilePath();
  final archiver = code.cCompiler?.archiver.toFilePath();
  final cxxCompiler = _deriveCppCompiler(compiler);
  final generator = _cmakeGenerator();
  final runtimeEnv = _runtimeBuildEnvironment(packageRootPath);
  final ortEnabled =
      _runtimeEnvValue(runtimeEnv, 'DART_INFERENCE_ENABLE_ORT') == '1';
  final ortInclude = _runtimeEnvValue(
    runtimeEnv,
    'DART_INFERENCE_ORT_INCLUDE_DIR',
  );
  final ortLibrary = _runtimeEnvValue(runtimeEnv, 'DART_INFERENCE_ORT_LIBRARY');
  final ortRuntimeLibrary = _runtimeEnvValue(
    runtimeEnv,
    'DART_INFERENCE_ORT_RUNTIME_LIBRARY',
  );
  final litertRuntimeLibrary =
      _runtimeEnvValue(runtimeEnv, 'DART_INFERENCE_LITERT_LIBRARY') ??
      _runtimeEnvValue(runtimeEnv, 'DART_INFERENCE_TFLITE_LIBRARY');
  final litertExtraRuntimeLibraries = _runtimeLibraryList(
    _runtimeEnvValue(runtimeEnv, 'DART_INFERENCE_LITERT_EXTRA_LIBRARIES'),
  );
  logger.info(
    'runtime-backend-env target=${code.targetOS.name} '
    'runtime=zig '
    'ortEnabled=$ortEnabled '
    'ortInclude=${ortInclude ?? '<unset>'} '
    'ortLibrary=${ortLibrary ?? '<unset>'} '
    'ortRuntime=${ortRuntimeLibrary ?? '<unset>'} '
    'litertRuntime=${litertRuntimeLibrary ?? '<unset>'} '
    'litertExtra=${litertExtraRuntimeLibraries.join(',')}',
  );
  final androidNdk = _resolveAndroidNdk();
  if (code.targetOS == OS.android && androidNdk == null) {
    throw StateError(
      'Android runtime build requires a valid Android NDK. Set '
      'ANDROID_NDK_HOME or ANDROID_NDK_ROOT, or install one under the Android '
      'SDK ndk directory.',
    );
  }
  final configureArgs = <String>[
    '-S',
    packageRoot.resolve('native/runtime').toFilePath(),
    '-B',
    buildDirectoryPath,
    '-G',
    generator,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$outputDirectoryPath',
    '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$outputDirectoryPath',
    '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$outputDirectoryPath',
    '-DDINF_ENABLE_ORT=${ortEnabled ? 'ON' : 'OFF'}',
    if (ortEnabled && ortInclude != null) '-DDINF_ORT_INCLUDE_DIR=$ortInclude',
    if (ortEnabled && ortLibrary != null) '-DDINF_ORT_LIBRARY=$ortLibrary',
    if (_isAppleTarget(code.targetOS)) ...[
      '-DCMAKE_OSX_ARCHITECTURES=${_appleArchitectureName(code.targetArchitecture)}',
      '-DCMAKE_OSX_DEPLOYMENT_TARGET=${_deploymentTarget(code)}',
      if (code.targetOS == OS.iOS) ...[
        '-DCMAKE_SYSTEM_NAME=iOS',
        '-DCMAKE_OSX_SYSROOT=${_iosSdkName(code.iOS.targetSdk)}',
        '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY',
      ] else ...[
        '-DCMAKE_OSX_SYSROOT=macosx',
      ],
    ],
    if (code.targetOS == OS.android) ...[
      '-DCMAKE_TOOLCHAIN_FILE=$androidNdk/build/cmake/android.toolchain.cmake',
      '-DANDROID_ABI=${_androidAbiName(code.targetArchitecture)}',
      '-DANDROID_PLATFORM=android-${code.android.targetNdkApi}',
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

  if (!File.fromUri(adapterLibraryFile).existsSync()) {
    throw StateError(
      'Expected native runtime adapter library was not produced: '
      '$adapterLibraryFile',
    );
  }

  await _bundleRuntimeDependency(
    logger,
    input: input,
    output: output,
    outputDirectory: outputDirectory,
    sourcePath: ortEnabled
        ? ortRuntimeLibrary ?? _dynamicLibraryPath(ortLibrary)
        : null,
    assetName: 'onnxruntime',
  );
  if (ortEnabled) {
    await _bundleOrtProviderDependencies(
      logger,
      input: input,
      output: output,
      outputDirectory: outputDirectory,
      runtimeLibraryPath: ortRuntimeLibrary ?? _dynamicLibraryPath(ortLibrary),
    );
  }
  await _bundleRuntimeDependency(
    logger,
    input: input,
    output: output,
    outputDirectory: outputDirectory,
    sourcePath: litertRuntimeLibrary,
    assetName: 'litert',
  );
  var litertExtraIndex = 0;
  for (final extraLibrary in litertExtraRuntimeLibraries) {
    if (_sameRuntimeLibrary(extraLibrary, litertRuntimeLibrary)) {
      continue;
    }
    litertExtraIndex += 1;
    await _bundleRuntimeDependency(
      logger,
      input: input,
      output: output,
      outputDirectory: outputDirectory,
      sourcePath: extraLibrary,
      assetName: 'litert_extra_$litertExtraIndex',
    );
  }

  output.assets.code.add(
    CodeAsset(
      package: input.packageName,
      name: '${input.packageName}_runtime_adapter_dependency',
      linkMode: DynamicLoadingBundled(),
      file: adapterLibraryFile,
    ),
  );
  await _buildZigRuntimeAsset(
    logger,
    input: input,
    output: output,
    packageRoot: packageRoot,
    packageRootPath: packageRootPath,
    code: code,
    libraryFile: runtimeLibraryFile,
    adapterLibraryDirectory: outputDirectoryPath,
  );
}

Future<void> _buildZigRuntimeAsset(
  Logger logger, {
  required BuildInput input,
  required BuildOutputBuilder output,
  required Uri packageRoot,
  required String packageRootPath,
  required CodeConfig code,
  required Uri libraryFile,
  required String adapterLibraryDirectory,
}) async {
  final zig = await _resolveZigExecutable(packageRootPath);
  final pinnedVersion = _pinnedZigVersion(packageRootPath);
  final actualVersion = await _zigVersion(zig);
  if (actualVersion != pinnedVersion) {
    throw StateError(
      'dart_inference Zig runtime requires Zig $pinnedVersion, '
      'but $zig reports $actualVersion. Set DART_INFERENCE_ZIG to a pinned Zig '
      '$pinnedVersion executable.',
    );
  }

  final source = packageRoot.resolve('native/zig_runtime/runtime.zig');
  final args = <String>[
    'build-lib',
    source.toFilePath(),
    '-dynamic',
    '-O',
    'ReleaseFast',
    '-lc',
    '-femit-bin=${libraryFile.toFilePath()}',
    '-fallow-shlib-undefined',
    '-fstrip',
    '-L$adapterLibraryDirectory',
    '-ldart_inference_runtime_adapter',
    '-rpath',
    _runtimeOriginRpath(code.targetOS),
    '--cache-dir',
    packageRoot.resolve('.dart_tool/zig-cache').toFilePath(),
    '--global-cache-dir',
    packageRoot.resolve('.dart_tool/zig-global-cache').toFilePath(),
    '-target',
    _zigTarget(code),
  ];
  logger.info(
    'Building Zig runtime with Zig $actualVersion for ${code.targetOS.name}/'
    '${code.targetArchitecture.name}',
  );
  await _runProcess(
    logger,
    code: code,
    executable: zig,
    arguments: args,
    workingDirectory: packageRootPath,
  );

  if (!File.fromUri(libraryFile).existsSync()) {
    throw StateError(
      'Expected Zig runtime native library was not produced: $libraryFile',
    );
  }

  output.assets.code.add(
    CodeAsset(
      package: input.packageName,
      name: '${input.packageName}_runtime_bindings_generated.dart',
      linkMode: DynamicLoadingBundled(),
      file: libraryFile,
    ),
  );
}

Future<void> _bundleRuntimeDependency(
  Logger logger, {
  required BuildInput input,
  required BuildOutputBuilder output,
  required Uri outputDirectory,
  required String? sourcePath,
  required String assetName,
}) async {
  if (sourcePath == null || sourcePath.isEmpty) {
    return;
  }
  final source = File(sourcePath);
  if (!source.existsSync()) {
    throw StateError('Native runtime dependency does not exist: $sourcePath');
  }
  final filename = source.uri.pathSegments.last;
  final destination = outputDirectory.resolve(filename);
  final destinationPath = destination.toFilePath();
  if (source.absolute.path != File(destinationPath).absolute.path) {
    logger.info('Bundling native runtime dependency $sourcePath');
    await source.copy(destinationPath);
  }
  output.dependencies.add(source.uri);
  output.assets.code.add(
    CodeAsset(
      package: input.packageName,
      name: '${input.packageName}_${assetName}_dependency',
      linkMode: DynamicLoadingBundled(),
      file: destination,
    ),
  );
  final aliases = await _bundleRuntimeDependencyAliases(
    logger,
    outputDirectory: outputDirectory,
    destinationPath: destinationPath,
  );
  var aliasIndex = 0;
  for (final alias in aliases) {
    aliasIndex += 1;
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: '${input.packageName}_${assetName}_dependency_alias_$aliasIndex',
        linkMode: DynamicLoadingBundled(),
        file: alias.uri,
      ),
    );
  }
}

Future<void> _bundleOrtProviderDependencies(
  Logger logger, {
  required BuildInput input,
  required BuildOutputBuilder output,
  required Uri outputDirectory,
  required String? runtimeLibraryPath,
}) async {
  if (runtimeLibraryPath == null || runtimeLibraryPath.isEmpty) {
    return;
  }
  final runtimeLibrary = File(runtimeLibraryPath);
  final directory = runtimeLibrary.parent;
  if (!directory.existsSync()) {
    return;
  }
  final providerLibraries = directory.listSync().whereType<File>().where((
    file,
  ) {
    final name = file.uri.pathSegments.last;
    return _isOrtProviderLibraryName(name);
  }).toList()..sort((a, b) => a.path.compareTo(b.path));
  var index = 0;
  for (final providerLibrary in providerLibraries) {
    index += 1;
    await _bundleRuntimeDependency(
      logger,
      input: input,
      output: output,
      outputDirectory: outputDirectory,
      sourcePath: providerLibrary.path,
      assetName: 'onnxruntime_provider_$index',
    );
  }
}

bool _isOrtProviderLibraryName(String name) {
  if (name.startsWith('libonnxruntime_providers') &&
      (name.endsWith('.so') || name.contains('.so.'))) {
    return true;
  }
  if (name.startsWith('libonnxruntime_providers') && name.endsWith('.dylib')) {
    return true;
  }
  final lower = name.toLowerCase();
  return lower.startsWith('onnxruntime_providers') && lower.endsWith('.dll');
}

Future<List<File>> _bundleRuntimeDependencyAliases(
  Logger logger, {
  required Uri outputDirectory,
  required String destinationPath,
}) async {
  final destination = File(destinationPath);
  final filename = destination.uri.pathSegments.last;
  final aliases = <String>[];
  final linuxSoname = RegExp(
    r'^(lib[^/]+\.so\.\d+)\.\d+(?:\.\d+)*$',
  ).firstMatch(filename);
  if (linuxSoname != null) {
    aliases.add(linuxSoname.group(1)!);
  }
  if (filename.startsWith('libonnxruntime.so.')) {
    aliases.add('libonnxruntime.so');
  }
  if (aliases.isEmpty) {
    return const [];
  }
  final created = <File>[];
  final directory = Directory.fromUri(outputDirectory);
  for (final aliasName in aliases.toSet()) {
    if (aliasName == filename) {
      continue;
    }
    final aliasPath = '${directory.path}/$aliasName';
    final aliasFile = File(aliasPath);
    final aliasLink = Link(aliasPath);
    if (aliasLink.existsSync()) {
      await aliasLink.delete();
    } else if (aliasFile.existsSync()) {
      await aliasFile.delete();
    }
    try {
      logger.info('Bundling native runtime dependency alias $aliasName');
      await aliasLink.create(filename);
      created.add(aliasFile);
    } on FileSystemException {
      await destination.copy(aliasPath);
      created.add(aliasFile);
    }
  }
  return created;
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

Future<Set<Uri>> _collectDependencies(Uri packageRoot) async {
  final dependencies = <Uri>{};
  final runtimeOverride = _runtimeEnvOverrideFile(packageRoot.toFilePath());
  if (runtimeOverride != null && runtimeOverride.existsSync()) {
    dependencies.add(runtimeOverride.uri);
  }
  for (final relativePath in const [
    'native',
    'third_party',
    '.zigversion',
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

bool _isAppleTarget(OS os) => os == OS.iOS || os == OS.macOS;

bool _supportsRuntime(OS os) =>
    os == OS.iOS ||
    os == OS.macOS ||
    os == OS.windows ||
    os == OS.linux ||
    os == OS.android;

String? _dynamicLibraryPath(String? path) {
  if (path == null || path.isEmpty) {
    return null;
  }
  final lower = path.toLowerCase();
  if (lower.endsWith('.dll') ||
      lower.endsWith('.dylib') ||
      lower.endsWith('.so') ||
      lower.contains('.so.')) {
    return path;
  }
  return null;
}

Map<String, String> _runtimeBuildEnvironment(String packageRootPath) {
  final values = <String, String>{};
  final overrideFile = _runtimeEnvOverrideFile(packageRootPath);
  if (overrideFile != null && overrideFile.existsSync()) {
    try {
      final decoded = jsonDecode(overrideFile.readAsStringSync());
      if (decoded is Map) {
        for (final entry in decoded.entries) {
          final key = entry.key;
          final value = entry.value;
          if (key is String && value is String && value.isNotEmpty) {
            values[key] = value;
          }
        }
      }
    } catch (_) {
      // Ignore malformed override files and fall back to process environment.
    }
  }
  return values;
}

File? _runtimeEnvOverrideFile(String packageRootPath) {
  final explicit = Platform.environment['DART_INFERENCE_RUNTIME_ENV_FILE'];
  if (explicit != null && explicit.isNotEmpty) {
    return File(explicit);
  }
  return File('$packageRootPath/.dart_inference_runtime_env.json');
}

String? _runtimeEnvValue(Map<String, String> fileEnv, String name) {
  final processValue = Platform.environment[name];
  if (processValue != null && processValue.isNotEmpty) {
    return processValue;
  }
  final fileValue = fileEnv[name];
  if (fileValue != null && fileValue.isNotEmpty) {
    return fileValue;
  }
  return null;
}

List<String> _runtimeLibraryList(String? raw) {
  if (raw == null || raw.isEmpty) {
    return const [];
  }
  final separators = Platform.isWindows
      ? RegExp(r'[;,\n\r]+')
      : RegExp(r'[:,;\n\r]+');
  final seen = <String>{};
  final result = <String>[];
  for (final part in raw.split(separators)) {
    final value = part.trim();
    if (value.isEmpty || !seen.add(value)) {
      continue;
    }
    result.add(value);
  }
  return result;
}

bool _sameRuntimeLibrary(String left, String? right) {
  if (right == null || right.isEmpty) {
    return false;
  }
  return File(left).absolute.path == File(right).absolute.path;
}

String? _resolveAndroidNdk() {
  for (final name in const ['ANDROID_NDK_HOME', 'ANDROID_NDK_ROOT']) {
    final value = Platform.environment[name];
    if (_isValidAndroidNdk(value)) {
      return value;
    }
  }

  final sdkRoots = <String?>[
    Platform.environment['ANDROID_HOME'],
    Platform.environment['ANDROID_SDK_ROOT'],
    if (Platform.environment['HOME'] case final home?)
      if (Platform.isMacOS) '$home/Library/Android/sdk',
    if (Platform.environment['HOME'] case final home?)
      if (Platform.isLinux) '$home/Android/Sdk',
  ];
  final seen = <String>{};
  final candidates = <String>[];
  for (final sdkRoot in sdkRoots) {
    if (sdkRoot == null || sdkRoot.isEmpty || !seen.add(sdkRoot)) {
      continue;
    }
    final ndkDir = Directory('$sdkRoot/ndk');
    if (!ndkDir.existsSync()) {
      continue;
    }
    for (final entity in ndkDir.listSync()) {
      if (entity is Directory && _isValidAndroidNdk(entity.path)) {
        candidates.add(entity.path);
      }
    }
  }
  candidates.sort(_compareAndroidNdkPaths);
  return candidates.isEmpty ? null : candidates.last;
}

bool _isValidAndroidNdk(String? path) {
  if (path == null || path.isEmpty) {
    return false;
  }
  return File('$path/build/cmake/android.toolchain.cmake').existsSync();
}

int _compareAndroidNdkPaths(String left, String right) {
  final leftRevision = _androidNdkRevision(left);
  final rightRevision = _androidNdkRevision(right);
  for (
    var i = 0;
    i < math.max(leftRevision.length, rightRevision.length);
    i++
  ) {
    final leftPart = i < leftRevision.length ? leftRevision[i] : 0;
    final rightPart = i < rightRevision.length ? rightRevision[i] : 0;
    if (leftPart != rightPart) {
      return leftPart.compareTo(rightPart);
    }
  }
  return left.compareTo(right);
}

List<int> _androidNdkRevision(String path) {
  final source = File('$path/source.properties');
  final text = source.existsSync() ? source.readAsStringSync() : path;
  final match = RegExp(r'Pkg\.Revision\s*=\s*([0-9.]+)').firstMatch(text);
  final version = match?.group(1) ?? path.split(Platform.pathSeparator).last;
  return RegExp(
    r'\d+',
  ).allMatches(version).map((match) => int.parse(match.group(0)!)).toList();
}

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

String _androidAbiName(Architecture architecture) => switch (architecture) {
  Architecture.arm => 'armeabi-v7a',
  Architecture.arm64 => 'arm64-v8a',
  Architecture.ia32 => 'x86',
  Architecture.x64 => 'x86_64',
  Architecture.riscv64 => 'riscv64',
  Architecture.riscv32 => throw UnsupportedError(
    'riscv32 is unsupported for Android.',
  ),
  Architecture() => throw UnsupportedError(
    'Unsupported Android architecture: ${architecture.name}',
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
  if (cCompiler.endsWith('gcc')) {
    final candidate = '${cCompiler.substring(0, cCompiler.length - 3)}g++';
    if (File(candidate).existsSync()) {
      return candidate;
    }
  }
  return cCompiler;
}

Future<String> _resolveZigExecutable(String packageRootPath) async {
  final explicit = Platform.environment['DART_INFERENCE_ZIG'];
  if (explicit != null && explicit.isNotEmpty) {
    return explicit;
  }
  final generic = Platform.environment['ZIG'];
  if (generic != null && generic.isNotEmpty) {
    return generic;
  }
  final local = _localPinnedZigExecutable(packageRootPath);
  if (local != null) {
    return local;
  }
  return 'zig';
}

String? _localPinnedZigExecutable(String packageRootPath) {
  final host = Platform.isWindows
      ? 'windows'
      : Platform.isMacOS
      ? 'macos'
      : Platform.isLinux
      ? 'linux'
      : null;
  if (host == null) {
    return null;
  }
  final arch = _hostZigArchitectureName();
  if (arch == null) {
    return null;
  }
  final version = _pinnedZigVersion(packageRootPath);
  final suffix = Platform.isWindows ? 'zig.exe' : 'zig';
  final candidate =
      '$packageRootPath/.dart_tool/zig/zig-$arch-$host-$version/$suffix';
  return File(candidate).existsSync() ? candidate : null;
}

String? _hostZigArchitectureName() {
  if (Platform.isWindows) {
    final raw = Platform.environment['PROCESSOR_ARCHITECTURE'] ?? '';
    final wow = Platform.environment['PROCESSOR_ARCHITEW6432'] ?? '';
    final value = (wow.isNotEmpty ? wow : raw).toLowerCase();
    if (value.contains('arm64') || value.contains('aarch64')) {
      return 'aarch64';
    }
    if (value.contains('86')) {
      return 'x86_64';
    }
    return null;
  }
  final result = Process.runSync('uname', ['-m']);
  if (result.exitCode != 0) {
    return null;
  }
  final value = result.stdout.toString().trim().toLowerCase();
  return switch (value) {
    'x86_64' || 'amd64' => 'x86_64',
    'arm64' || 'aarch64' => 'aarch64',
    'armv7l' || 'armv7' || 'arm' => 'arm',
    'riscv64' => 'riscv64',
    _ => null,
  };
}

String _pinnedZigVersion(String packageRootPath) {
  final versionFile = File('$packageRootPath/.zigversion');
  if (versionFile.existsSync()) {
    final value = versionFile.readAsStringSync().trim();
    if (value.isNotEmpty) {
      return value;
    }
  }
  final toolchainFile = File(
    '$packageRootPath/native/zig_runtime/toolchain.json',
  );
  if (!toolchainFile.existsSync()) {
    throw StateError(
      'Missing Zig version metadata. Expected .zigversion or '
      'native/zig_runtime/toolchain.json.',
    );
  }
  final payload = jsonDecode(toolchainFile.readAsStringSync());
  if (payload is Map && payload['version'] is String) {
    final value = (payload['version'] as String).trim();
    if (value.isNotEmpty) {
      return value;
    }
  }
  throw StateError(
    'native/zig_runtime/toolchain.json does not contain a Zig version.',
  );
}

Future<String> _zigVersion(String executable) async {
  final result = await Process.run(executable, ['version']);
  if (result.exitCode != 0) {
    throw ProcessException(
      executable,
      const ['version'],
      result.stderr.toString(),
      result.exitCode,
    );
  }
  return result.stdout.toString().trim();
}

String _zigTarget(CodeConfig code) {
  final arch = _zigArchitectureName(code.targetArchitecture);
  return switch (code.targetOS) {
    OS.linux => '$arch-linux-gnu',
    OS.android =>
      arch == 'arm' ? 'arm-linux-androideabi' : '$arch-linux-android',
    OS.windows => '$arch-windows-gnu',
    OS.macOS => '$arch-macos',
    OS.iOS =>
      code.iOS.targetSdk == IOSSdk.iPhoneSimulator
          ? '$arch-ios-simulator'
          : '$arch-ios',
    OS() => throw UnsupportedError(
      'Unsupported Zig runtime target OS: ${code.targetOS.name}',
    ),
  };
}

String _runtimeOriginRpath(OS os) => switch (os) {
  OS.iOS || OS.macOS => '@loader_path',
  OS.linux || OS.android => r'$ORIGIN',
  OS.windows => '.',
  OS() => throw UnsupportedError(
    'Unsupported runtime rpath target OS: ${os.name}',
  ),
};

String _zigArchitectureName(Architecture architecture) =>
    switch (architecture) {
      Architecture.arm => 'arm',
      Architecture.arm64 => 'aarch64',
      Architecture.ia32 => 'x86',
      Architecture.x64 => 'x86_64',
      Architecture.riscv64 => 'riscv64',
      Architecture.riscv32 => throw UnsupportedError(
        'riscv32 is unsupported for Zig runtime builds.',
      ),
      Architecture() => throw UnsupportedError(
        'Unsupported Zig runtime architecture: ${architecture.name}',
      ),
    };

String _cmakeGenerator() {
  if (_executableExists('ninja') || _executableExists('ninja-build')) {
    return 'Ninja';
  }
  return Platform.isWindows ? 'NMake Makefiles' : 'Unix Makefiles';
}

bool _executableExists(String executable) {
  final path = Platform.environment['PATH'];
  if (path == null || path.isEmpty) {
    return false;
  }
  final extensions = Platform.isWindows
      ? (Platform.environment['PATHEXT'] ?? '.exe;.bat;.cmd')
            .split(';')
            .where((value) => value.isNotEmpty)
            .toList(growable: false)
      : const [''];
  for (final directory in path.split(Platform.isWindows ? ';' : ':')) {
    if (directory.isEmpty) {
      continue;
    }
    for (final extension in extensions) {
      final candidate =
          '$directory${Platform.pathSeparator}$executable$extension';
      if (File(candidate).existsSync()) {
        return true;
      }
    }
  }
  return false;
}

String _cmdQuote(String value) => '"${value.replaceAll('"', '""')}"';
