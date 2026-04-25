import CoreML
import CoreMLLLM
import Foundation
import ImageIO

#if canImport(Darwin)
import Darwin.Mach
#endif

@main
struct CoreMLLLMBaseline {
  static func main() async {
    do {
      let args = try Args(CommandLine.arguments)
      if args.flag("help") {
        print(usage)
        return
      }
      let report = try await run(args)
      let data = try JSONSerialization.data(
        withJSONObject: report,
        options: [.prettyPrinted, .sortedKeys]
      )
      let text = String(decoding: data, as: UTF8.self) + "\n"
      if let out = args.option("out") {
        let url = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(
          at: url.deletingLastPathComponent(),
          withIntermediateDirectories: true,
          attributes: nil
        )
        try text.write(to: url, atomically: false, encoding: .utf8)
      }
      print(text, terminator: "")
    } catch {
      FileHandle.standardError.write(Data("coreml-llm baseline failed: \(error)\n".utf8))
      exit(2)
    }
  }

  private static func run(_ args: Args) async throws -> [String: Any] {
    let task = args.option("task") ?? "text"
    switch task {
    case "text":
      return try await runText(args)
    case "function":
      return try await runFunction(args)
    case "embedding":
      return try await runEmbedding(args)
    case "vlm":
      return try await runVlm(args)
    default:
      throw ArgumentError("Unsupported --task \(task)")
    }
  }

  private static func runText(_ args: Args) async throws -> [String: Any] {
    let modelId = try args.required("model-id")
    let artifact = try args.required("artifact")
    let platform = args.option("platform") ?? "ios"
    let prompt = try readPrompt(args)
    let warmup = args.int("warmup") ?? 1
    let iters = args.int("iters") ?? 5
    let maxTokens = args.int("max-tokens") ?? 64
    let units = computeUnits(args.option("compute-units") ?? "cpuAndNeuralEngine")

    let modelDirectory = URL(fileURLWithPath: artifact)
    let sampler = PeakMemorySampler()
    sampler.start()
    let llm = try await CoreMLLLM.load(from: modelDirectory, computeUnits: units)

    for _ in 0..<warmup {
      _ = try await runOnce(llm: llm, prompt: prompt, maxTokens: maxTokens)
      sampler.sample()
    }

    var records: [RunRecord] = []
    for _ in 0..<max(1, iters) {
      records.append(try await runOnce(llm: llm, prompt: prompt, maxTokens: maxTokens))
      sampler.sample()
    }
    let memory = sampler.stop()
    let last = records.last!
    let count = Double(records.count)
    let endToEndMs = records.reduce(0.0) { $0 + $1.endToEndMs } / count
    let ttftMs = records.reduce(0.0) { $0 + $1.ttftMs } / count
    let decodeTps = records.reduce(0.0) { $0 + $1.decodeTokensPerSecond } / count

    return [
      "model_id": modelId,
      "platform": platform,
      "engine": "coreml-llm",
      "artifact": artifact,
      "correctness": [
        "output_text": last.text,
        "token_ids": last.tokenIds,
        "prompt_token_ids": last.promptTokenIds,
        "generated_token_count": last.tokenIds.count,
      ],
      "metrics": [
        "decode_tokens_per_second": decodeTps,
        "ttft_ms": ttftMs,
        "end_to_end_ms": endToEndMs,
        "peak_memory_bytes": memory.peakMemoryBytes,
      ],
      "device_profile": memory.deviceProfile(
        runtime: "CoreML-LLM",
        computeUnits: computeUnitsName(units)
      ),
    ]
  }

  private static func runFunction(_ args: Args) async throws -> [String: Any] {
    let modelId = try args.required("model-id")
    let artifact = try args.required("artifact")
    let platform = args.option("platform") ?? "ios"
    let prompt = try readPrompt(args)
    let warmup = args.int("warmup") ?? 1
    let iters = args.int("iters") ?? 5
    let maxTokens = args.int("max-tokens") ?? 64
    let units = computeUnits(args.option("compute-units") ?? "cpuAndNeuralEngine")
    let tools = try readTools(args)

    let sampler = PeakMemorySampler()
    sampler.start()
    let model = try await FunctionGemma.load(
      bundleURL: URL(fileURLWithPath: artifact),
      computeUnits: units
    )

    for _ in 0..<warmup {
      _ = try runFunctionOnce(
        model: model,
        prompt: prompt,
        tools: tools,
        maxTokens: maxTokens
      )
      sampler.sample()
    }

    var records: [FunctionRunRecord] = []
    for _ in 0..<max(1, iters) {
      records.append(
        try runFunctionOnce(
          model: model,
          prompt: prompt,
          tools: tools,
          maxTokens: maxTokens
        )
      )
      sampler.sample()
    }
    let memory = sampler.stop()
    let last = records.last!
    let count = Double(records.count)
    let endToEndMs = records.reduce(0.0) { $0 + $1.endToEndMs } / count

    var correctness: [String: Any] = [
      "output_text": last.text,
      "generated_token_count": last.tokenCount,
    ]
    if let functionCall = last.functionCall {
      correctness["function_call"] = functionCall
    }

    return [
      "model_id": modelId,
      "platform": platform,
      "engine": "coreml-llm",
      "artifact": artifact,
      "correctness": correctness,
      "metrics": [
        "decode_tokens_per_second": records.reduce(0.0) { $0 + $1.tokensPerSecond } / count,
        "end_to_end_ms": endToEndMs,
        "peak_memory_bytes": memory.peakMemoryBytes,
      ],
      "device_profile": memory.deviceProfile(
        runtime: "CoreML-LLM FunctionGemma",
        computeUnits: computeUnitsName(units)
      ),
    ]
  }

  private static func runEmbedding(_ args: Args) async throws -> [String: Any] {
    let modelId = try args.required("model-id")
    let artifact = try args.required("artifact")
    let platform = args.option("platform") ?? "ios"
    let query = try readEmbeddingQuery(args)
    let warmup = args.int("warmup") ?? 1
    let iters = args.int("iters") ?? 5
    let dim = args.int("embedding-dim")
    let units = computeUnits(args.option("compute-units") ?? "cpuAndNeuralEngine")

    let sampler = PeakMemorySampler()
    sampler.start()
    let model = try await EmbeddingGemma.load(
      bundleURL: URL(fileURLWithPath: artifact),
      computeUnits: units
    )

    for _ in 0..<warmup {
      _ = try model.encode(text: query, task: nil, dim: dim)
      sampler.sample()
    }

    var vectors: [[Float]] = []
    var times: [Double] = []
    for _ in 0..<max(1, iters) {
      let start = DispatchTime.now().uptimeNanoseconds
      let vector = try model.encode(text: query, task: nil, dim: dim)
      let end = DispatchTime.now().uptimeNanoseconds
      vectors.append(vector)
      times.append(millis(from: start, to: end))
      sampler.sample()
    }
    let memory = sampler.stop()
    let vector = vectors.last ?? []
    let endToEndMs = times.reduce(0.0, +) / Double(max(1, times.count))

    return [
      "model_id": modelId,
      "platform": platform,
      "engine": "coreml-llm",
      "artifact": artifact,
      "correctness": [
        "embedding_dim": vector.count,
        "output_values": [
          "embedding": [
            "dtype": "float32",
            "shape": [vector.count],
            "values": vector,
          ],
        ],
      ],
      "metrics": [
        "end_to_end_ms": endToEndMs,
        "peak_memory_bytes": memory.peakMemoryBytes,
      ],
      "device_profile": memory.deviceProfile(
        runtime: "CoreML-LLM EmbeddingGemma",
        computeUnits: computeUnitsName(units)
      ),
    ]
  }

  private static func runVlm(_ args: Args) async throws -> [String: Any] {
    let modelId = try args.required("model-id")
    let artifact = try args.required("artifact")
    let platform = args.option("platform") ?? "ios"
    let prompt = try readPrompt(args)
    let image = try readImage(args)
    let warmup = args.int("warmup") ?? 1
    let iters = args.int("iters") ?? 5
    let maxTokens = args.int("max-tokens") ?? 64
    let units = computeUnits(args.option("compute-units") ?? "cpuAndNeuralEngine")

    let sampler = PeakMemorySampler()
    sampler.start()
    let llm = try await CoreMLLLM.load(
      from: URL(fileURLWithPath: artifact),
      computeUnits: units
    )

    for _ in 0..<warmup {
      _ = try await runOnce(
        llm: llm,
        prompt: prompt,
        image: image,
        maxTokens: maxTokens
      )
      sampler.sample()
    }

    var records: [RunRecord] = []
    for _ in 0..<max(1, iters) {
      records.append(
        try await runOnce(
          llm: llm,
          prompt: prompt,
          image: image,
          maxTokens: maxTokens
        )
      )
      sampler.sample()
    }
    let memory = sampler.stop()
    let last = records.last!
    let count = Double(records.count)

    return [
      "model_id": modelId,
      "platform": platform,
      "engine": "coreml-llm",
      "artifact": artifact,
      "correctness": [
        "output_text": last.text,
        "token_ids": last.tokenIds,
        "prompt_token_ids": last.promptTokenIds,
        "generated_token_count": last.tokenIds.count,
      ],
      "metrics": [
        "decode_tokens_per_second": records.reduce(0.0) { $0 + $1.decodeTokensPerSecond } / count,
        "ttft_ms": records.reduce(0.0) { $0 + $1.ttftMs } / count,
        "end_to_end_ms": records.reduce(0.0) { $0 + $1.endToEndMs } / count,
        "peak_memory_bytes": memory.peakMemoryBytes,
      ],
      "device_profile": memory.deviceProfile(
        runtime: "CoreML-LLM VLM",
        computeUnits: computeUnitsName(units)
      ),
    ]
  }

  private static func runOnce(
    llm: CoreMLLLM,
    prompt: String,
    image: CGImage? = nil,
    maxTokens: Int
  ) async throws -> RunRecord {
    let start = DispatchTime.now().uptimeNanoseconds
    var firstToken: UInt64?
    var text = ""
    let stream = try await llm.stream(prompt, image: image, maxTokens: maxTokens)
    for await token in stream {
      if firstToken == nil {
        firstToken = DispatchTime.now().uptimeNanoseconds
      }
      text += token
    }
    let end = DispatchTime.now().uptimeNanoseconds
    let first = firstToken ?? end
    return RunRecord(
      text: text,
      tokenIds: llm.lastEmittedTokenIDs.map(Int.init),
      promptTokenIds: llm.lastPromptTokenIDs.map(Int.init),
      endToEndMs: millis(from: start, to: end),
      ttftMs: millis(from: start, to: first),
      decodeTokensPerSecond: llm.tokensPerSecond
    )
  }

  private static func readPrompt(_ args: Args) throws -> String {
    if let value = args.option("prompt") {
      return value
    }
    if let path = args.option("prompt-file") {
      return try String(contentsOfFile: path, encoding: .utf8)
    }
    throw ArgumentError("Missing --prompt or --prompt-file")
  }

  private static func readEmbeddingQuery(_ args: Args) throws -> String {
    if let value = args.option("embedding-query") {
      return value
    }
    if let path = args.option("embedding-query-file") {
      return try String(contentsOfFile: path, encoding: .utf8)
    }
    return try readPrompt(args)
  }

  private static func readTools(_ args: Args) throws -> [[String: Any]]? {
    let data: Data
    if let value = args.option("tools-json") {
      data = Data(value.utf8)
    } else if let path = args.option("tools-file") {
      data = try Data(contentsOf: URL(fileURLWithPath: path))
    } else {
      return nil
    }
    let decoded = try JSONSerialization.jsonObject(with: data)
    guard let tools = decoded as? [[String: Any]] else {
      if let tool = decoded as? [String: Any] {
        return [normalizeTool(tool)]
      }
      throw ArgumentError("Tool schema must be a JSON object or array")
    }
    return tools.map(normalizeTool)
  }

  private static func normalizeTool(_ tool: [String: Any]) -> [String: Any] {
    if tool["type"] != nil || tool["function"] != nil {
      return tool
    }
    return [
      "type": "function",
      "function": tool,
    ]
  }

  private static func readImage(_ args: Args) throws -> CGImage {
    guard let path = args.option("image-file") else {
      throw ArgumentError("Missing --image-file for --task vlm")
    }
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      throw ArgumentError("Unable to decode image: \(path)")
    }
    return image
  }

  private static func computeUnits(_ value: String) -> MLComputeUnits {
    switch value {
    case "cpuOnly":
      return .cpuOnly
    case "cpuAndGPU":
      return .cpuAndGPU
    case "all":
      return .all
    case "cpuAndNeuralEngine", "ane":
      return .cpuAndNeuralEngine
    default:
      return .cpuAndNeuralEngine
    }
  }

  private static func computeUnitsName(_ units: MLComputeUnits) -> String {
    switch units {
    case .cpuOnly:
      return "cpuOnly"
    case .cpuAndGPU:
      return "cpuAndGPU"
    case .all:
      return "all"
    case .cpuAndNeuralEngine:
      return "cpuAndNeuralEngine"
    @unknown default:
      return "unknown"
    }
  }

  static func millis(from start: UInt64, to end: UInt64) -> Double {
    Double(end - start) / 1_000_000.0
  }
}

struct RunRecord {
  let text: String
  let tokenIds: [Int]
  let promptTokenIds: [Int]
  let endToEndMs: Double
  let ttftMs: Double
  let decodeTokensPerSecond: Double
}

struct FunctionRunRecord {
  let text: String
  let functionCall: String?
  let tokenCount: Int
  let endToEndMs: Double
  let tokensPerSecond: Double
}

func runFunctionOnce(
  model: FunctionGemma,
  prompt: String,
  tools: [[String: Any]]?,
  maxTokens: Int
) throws -> FunctionRunRecord {
  let start = DispatchTime.now().uptimeNanoseconds
  var tokenCount = 0
  let text = try model.generate(
    messages: [["role": "user", "content": prompt]],
    tools: tools,
    maxNewTokens: maxTokens
  ) { _ in
    tokenCount += 1
    return true
  }
  let end = DispatchTime.now().uptimeNanoseconds
  let elapsedMs = CoreMLLLMBaseline.millis(from: start, to: end)
  let emittedTokenCount = max(1, tokenCount)
  return FunctionRunRecord(
    text: text,
    functionCall: model.extractFunctionCall(from: text),
    tokenCount: emittedTokenCount,
    endToEndMs: elapsedMs,
    tokensPerSecond: Double(emittedTokenCount) / max(elapsedMs / 1000.0, 0.001)
  )
}

final class PeakMemorySampler {
  private let queue = DispatchQueue(label: "dart_mlx_ffi.coreml_llm.peak_memory")
  private var timer: DispatchSourceTimer?
  private var peak: UInt64 = 0

  func start() {
    sample()
    let source = DispatchSource.makeTimerSource(queue: queue)
    source.schedule(deadline: .now(), repeating: .milliseconds(10))
    source.setEventHandler { [weak self] in
      self?.record(MemorySnapshot.current().peakMemoryBytes)
    }
    timer = source
    source.resume()
  }

  func sample() {
    let value = MemorySnapshot.current().peakMemoryBytes
    queue.sync {
      record(value)
    }
  }

  func stop() -> MemorySnapshot {
    timer?.cancel()
    timer = nil
    let snapshot = MemorySnapshot.current()
    queue.sync {
      record(snapshot.peakMemoryBytes)
    }
    let recordedPeak = queue.sync { peak }
    return snapshot.withPeak(max(snapshot.peakMemoryBytes, recordedPeak))
  }

  private func record(_ value: UInt64) {
    if value > peak {
      peak = value
    }
  }
}

struct MemorySnapshot {
  let peakMemoryBytes: UInt64
  let rawPeakMemoryField: String
  let physFootprint: UInt64?
  let residentSize: UInt64?
  let virtualSize: UInt64?

  func withPeak(_ peak: UInt64) -> MemorySnapshot {
    MemorySnapshot(
      peakMemoryBytes: peak,
      rawPeakMemoryField: rawPeakMemoryField,
      physFootprint: physFootprint,
      residentSize: residentSize,
      virtualSize: virtualSize
    )
  }

  func deviceProfile(runtime: String, computeUnits: String) -> [String: Any] {
    var profile: [String: Any] = [
      "runtime": runtime,
      "compute_units": computeUnits,
      "raw_peak_memory_field": rawPeakMemoryField,
    ]
    if let physFootprint {
      profile["phys_footprint"] = physFootprint
    }
    if let residentSize {
      profile["resident_size"] = residentSize
    }
    if let virtualSize {
      profile["virtual_size"] = virtualSize
    }
    return profile
  }

  static func current() -> MemorySnapshot {
    #if canImport(Darwin)
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    if result == KERN_SUCCESS {
      let footprint = UInt64(info.phys_footprint)
      return MemorySnapshot(
        peakMemoryBytes: footprint,
        rawPeakMemoryField: "phys_footprint",
        physFootprint: footprint,
        residentSize: UInt64(info.resident_size),
        virtualSize: UInt64(info.virtual_size)
      )
    }
    #endif
    return MemorySnapshot(
      peakMemoryBytes: 0,
      rawPeakMemoryField: "unavailable",
      physFootprint: nil,
      residentSize: nil,
      virtualSize: nil
    )
  }
}

struct Args {
  private let values: [String: String?]

  init(_ raw: [String]) throws {
    var parsed: [String: String?] = [:]
    var index = 1
    while index < raw.count {
      let arg = raw[index]
      guard arg.hasPrefix("--") else {
        throw ArgumentError("Unexpected positional argument: \(arg)")
      }
      let name = String(arg.dropFirst(2))
      if index + 1 < raw.count && !raw[index + 1].hasPrefix("--") {
        parsed[name] = raw[index + 1]
        index += 2
      } else {
        parsed[name] = nil
        index += 1
      }
    }
    values = parsed
  }

  func flag(_ name: String) -> Bool {
    values.keys.contains(name)
  }

  func option(_ name: String) -> String? {
    values[name] ?? nil
  }

  func required(_ name: String) throws -> String {
    guard let value = option(name), !value.isEmpty else {
      throw ArgumentError("Missing --\(name)")
    }
    return value
  }

  func int(_ name: String) -> Int? {
    guard let value = option(name) else {
      return nil
    }
    return Int(value)
  }
}

struct ArgumentError: Error, CustomStringConvertible {
  let description: String

  init(_ description: String) {
    self.description = description
  }
}

let usage = """
Usage:
  swift run coreml-llm-baseline \\
    --model-id <id> \\
    --artifact <CoreML-LLM model directory> \\
    [--task text|function|embedding|vlm] \\
    (--prompt <text> | --prompt-file <path>) \\
    [--platform ios] [--warmup 1] [--iters 5] [--max-tokens 64] \\
    [--tools-file tools.json] [--embedding-query-file query.txt] \\
    [--image-file image.png] [--embedding-dim 768] \\
    [--compute-units cpuAndNeuralEngine] [--out report.json]
"""
