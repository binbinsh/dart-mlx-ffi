// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "DartInferenceCoreMLLLMBaseline",
  platforms: [
    .iOS(.v18),
    .macOS(.v15),
  ],
  products: [
    .executable(name: "coreml-llm-baseline", targets: ["CoreMLLLMBaseline"]),
  ],
  dependencies: [
    .package(url: "https://github.com/john-rocky/CoreML-LLM", from: "1.3.0"),
  ],
  targets: [
    .executableTarget(
      name: "CoreMLLLMBaseline",
      dependencies: [
        .product(name: "CoreMLLLM", package: "CoreML-LLM"),
      ],
      swiftSettings: [.swiftLanguageMode(.v5)]
    ),
  ]
)
