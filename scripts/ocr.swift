// Apple Vision フレームワークで画像からテキストを抽出する簡易OCRツール。
// 使い方: swift ocr.swift <画像ファイル1> [画像ファイル2] ...
// 各画像について、認識したテキストを標準出力に書き出す。

import Vision
import AppKit

func ocr(path: String) -> String {
    guard let image = NSImage(contentsOfFile: path),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return "[画像読み込み失敗: \(path)]"
    }

    var resultText = ""
    let semaphore = DispatchSemaphore(value: 0)

    let request = VNRecognizeTextRequest { request, error in
        defer { semaphore.signal() }
        guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
        let lines = observations.compactMap { $0.topCandidates(1).first?.string }
        resultText = lines.joined(separator: "\n")
    }
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["ja-JP", "en-US"]
    request.usesLanguageCorrection = true

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try? handler.perform([request])
    semaphore.wait()

    return resultText
}

let args = CommandLine.arguments.dropFirst()
for path in args {
    print("===== \(path) =====")
    print(ocr(path: path))
    print("")
}
