// Core AI runtime bridge. The Python wheel currently omits expectFrequentReshapes.
import Foundation
import CoreAI

struct Request: Decodable {
    let ids: [Int32]
    let total: Int
    let state: Int
    let function: String
    let output: String
}

@main struct RuntimeBridge {
    static func main() async throws {
        let args = CommandLine.arguments
        let shape = try JSONDecoder().decode([Int].self, from: Data(args[2].utf8))
        let precision = args[3]
        guard precision == "float16" || precision == "float32" else {
            throw NSError(domain: "aguardente", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Runtime bridge supports float16/float32 caches"])
        }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model = try await AIModel(contentsOf: URL(fileURLWithPath: args[1]), options: options)
        let ready = try JSONSerialization.data(withJSONObject: ["functions": model.functionNames])
        FileHandle.standardOutput.write(ready + Data([10]))
        var caches: [Int: (NDArray, NDArray)] = [:]
        let count = shape.reduce(1, *)
        while let line = readLine() {
            let request = try JSONDecoder().decode(Request.self, from: Data(line.utf8))
            if request.function == "__release__" {
                caches.removeValue(forKey: request.state)
                FileHandle.standardOutput.write(Data("{}\n".utf8))
                continue
            }
            guard !request.ids.isEmpty, request.total >= request.ids.count,
                  request.total <= shape[3] else {
                throw NSError(domain: "aguardente", code: 3,
                              userInfo: [NSLocalizedDescriptionKey: "Invalid sequence/cache bounds"])
            }
            guard let function = try model.loadFunction(named: request.function) else {
                throw NSError(domain: "aguardente", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "Missing function \(request.function)"])
            }
            let previous = caches.removeValue(forKey: request.state)
            func zero() -> NDArray {
                if precision == "float32" {
                    return NDArray(scalars: Array(repeating: Float(0), count: count), shape: shape)
                }
                return NDArray(scalars: Array(repeating: Float16(0), count: count), shape: shape)
            }
            var key = previous?.0 ?? zero()
            var value = previous?.1 ?? zero()
            var states = InferenceFunction.MutableViews()
            states.insert(&key, for: "keyCache")
            states.insert(&value, for: "valueCache")
            var outputs = try await function.run(inputs: [
                "input_ids": NDArray(scalars: request.ids, shape: [1, request.ids.count]),
                "position_ids": NDArray(scalars: (0..<request.total).map(Int32.init), shape: [1, request.total])
            ], states: states)
            caches[request.state] = (key, value)
            var response: [String: Any] = [:]
            if let logits = outputs.remove("logits")?.ndArray {
                var scalars: [Float] = []
                if logits.scalarType == .float32 {
                    let view = logits.view(as: Float.self)
                    guard let span = view.contiguousElements else { fatalError("Noncontiguous logits") }
                    for i in 0..<span.count { scalars.append(span[i]) }
                } else {
                    let view = logits.view(as: Float16.self)
                    guard let span = view.contiguousElements else { fatalError("Noncontiguous logits") }
                    for i in 0..<span.count { scalars.append(Float(span[i])) }
                }
                try scalars.withUnsafeBytes { try Data($0).write(to: URL(fileURLWithPath: request.output)) }
                response["shape"] = logits.shape
            }
            let data = try JSONSerialization.data(withJSONObject: response, options: [.sortedKeys])
            FileHandle.standardOutput.write(data + Data([10]))
        }
    }
}
