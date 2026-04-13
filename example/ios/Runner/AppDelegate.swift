import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private let buildMarker = "launch-bridge-v17-native-argmax"

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    application.isIdleTimerDisabled = true
    writeBootMarker(stage: "didFinishLaunching")
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  override func applicationDidBecomeActive(_ application: UIApplication) {
    super.applicationDidBecomeActive(application)
    writeBootMarker(stage: "didBecomeActive")
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
  }

  private func writeBootMarker(stage: String) {
    guard
      let docs = FileManager.default.urls(
        for: .documentDirectory,
        in: .userDomainMask
      ).first
    else {
      return
    }

    let reportsDir = docs.appendingPathComponent("paddle_ocr_vl_reports", isDirectory: true)
    let markerPath = reportsDir.appendingPathComponent("native_boot.log")
    let formatter = ISO8601DateFormatter()
    let bundleURL = Bundle.main.bundleURL.lastPathComponent
    let env = ProcessInfo.processInfo.environment
    let rawTarget = env["POCR_TARGET_CASES"] ?? ""
    let rawReset = env["POCR_RESET_CASES"] ?? ""
    let target = rawTarget.isEmpty ? "-" : rawTarget
    let reset = rawReset.isEmpty ? "-" : rawReset
    let line =
      "\(formatter.string(from: Date())) stage=\(stage) build=\(buildMarker) bundle=\(bundleURL) target=\(target) reset=\(reset)\n"
    let launchConfigPath = reportsDir.appendingPathComponent("launch_env.json")
    let launchConfig: [String: String] = [
      "build_marker": buildMarker,
      "target_cases": rawTarget,
      "reset_cases": rawReset,
      "stage": stage,
      "timestamp": formatter.string(from: Date()),
    ]

    do {
      try FileManager.default.createDirectory(
        at: reportsDir,
        withIntermediateDirectories: true
      )
      let launchConfigData = try JSONSerialization.data(withJSONObject: launchConfig)
      try launchConfigData.write(to: launchConfigPath, options: .atomic)
      if FileManager.default.fileExists(atPath: markerPath.path) {
        let handle = try FileHandle(forWritingTo: markerPath)
        defer { try? handle.close() }
        handle.seekToEndOfFile()
        if let data = line.data(using: .utf8) {
          handle.write(data)
        }
      } else {
        try line.write(to: markerPath, atomically: true, encoding: .utf8)
      }
    } catch {
      NSLog("native_boot write failed: %@", String(describing: error))
    }
  }
}
