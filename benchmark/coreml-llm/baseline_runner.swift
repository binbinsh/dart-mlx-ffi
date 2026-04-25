import Foundation

func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data((message + "\n").utf8))
  exit(2)
}

let packageDirectory = URL(fileURLWithPath: #filePath)
  .deletingLastPathComponent()
  .appendingPathComponent("swift_baseline")

if CommandLine.arguments.contains("--help") {
  print(
    """
    Usage:
      swift benchmark/coreml-llm/baseline_runner.swift \\
        --model-id <id> \\
        --artifact <CoreML-LLM model directory> \\
        [--task text|function|embedding|vlm] \\
        (--prompt <text> | --prompt-file <path>) \\
        [--platform ios] [--warmup 1] [--iters 5] [--max-tokens 64] \\
        [--tools-file tools.json] [--embedding-query-file query.txt] \\
        [--image-file image.png] [--embedding-dim 768] \\
        [--compute-units cpuAndNeuralEngine] [--out report.json]
    """
  )
  exit(0)
}

let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
process.arguments = [
  "swift",
  "run",
  "--package-path",
  packageDirectory.path,
  "coreml-llm-baseline",
  "--",
] + Array(CommandLine.arguments.dropFirst())
process.standardInput = FileHandle.standardInput
process.standardOutput = FileHandle.standardOutput
process.standardError = FileHandle.standardError

do {
  try process.run()
  process.waitUntilExit()
  exit(process.terminationStatus)
} catch {
  fail("Failed to launch SwiftPM CoreML-LLM baseline runner: \(error)")
}
