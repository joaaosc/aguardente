import Foundation
import CoreAI
import Darwin

struct TensorFile: Codable {
    let path: String
    let shape: [Int]
    let dtype: String
}
struct TaskRequest: Decodable {
    let inputs: [String: TensorFile]
    let outputs: [String]
    let directory: String
}
func failure(_ text: String) -> NSError {
    NSError(domain: "aguardente", code: 1, userInfo: [NSLocalizedDescriptionKey: text])
}
func readScalars<T>(_ file: TensorFile, _: T.Type) throws -> [T] {
    let data = try Data(contentsOf: URL(fileURLWithPath: file.path))
    let count = file.shape.reduce(1, *)
    guard count > 0, data.count == count * MemoryLayout<T>.stride else {
        throw failure("Invalid tensor size")
    }
    return data.withUnsafeBytes { raw in
        (0..<count).map { raw.loadUnaligned(fromByteOffset: $0 * MemoryLayout<T>.stride, as: T.self) }
    }
}
@main struct TaskRuntime {
    static func main() async throws {
        let options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        let started = Date()
        let model = try await AIModel(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]), options: options)
        guard let function = try model.loadFunction(named: "main") else { throw failure("Missing main") }
        let ready = try JSONSerialization.data(withJSONObject: ["functions": model.functionNames,
                                                                 "load_seconds": Date().timeIntervalSince(started)])
        FileHandle.standardOutput.write(ready + Data([10]))
        while let line = readLine() {
            let req = try JSONDecoder().decode(TaskRequest.self, from: Data(line.utf8))
            var inputs: [String: NDArray] = [:]
            for (name, file) in req.inputs {
                switch file.dtype {
                case "float32": inputs[name] = NDArray(scalars: try readScalars(file, Float.self), shape: file.shape)
                case "float16": inputs[name] = NDArray(scalars: try readScalars(file, Float16.self), shape: file.shape)
                case "int32": inputs[name] = NDArray(scalars: try readScalars(file, Int32.self), shape: file.shape)
                default: throw failure("Unsupported input dtype: \(file.dtype)")
                }
            }
            let tick = Date()
            var outputs = try await function.run(inputs: inputs)
            let seconds = Date().timeIntervalSince(tick)
            var result: [String: TensorFile] = [:]
            for (index, name) in req.outputs.enumerated() {
                guard let array = outputs.remove(name)?.ndArray else { throw failure("Missing output: \(name)") }
                let path = URL(fileURLWithPath: req.directory).appendingPathComponent("out-\(index).bin")
                var values: [Float] = []
                switch array.scalarType {
                case .float32:
                    let view = array.view(as: Float.self)
                    guard let span = view.contiguousElements else { throw failure("Noncontiguous output") }
                    for i in 0..<span.count { values.append(span[i]) }
                case .float16:
                    let view = array.view(as: Float16.self)
                    guard let span = view.contiguousElements else { throw failure("Noncontiguous output") }
                    for i in 0..<span.count { values.append(Float(span[i])) }
                default: throw failure("Unsupported output dtype")
                }
                try values.withUnsafeBytes { try Data($0).write(to: path) }
                result[name] = TensorFile(path: path.path, shape: array.shape, dtype: "float32")
            }
            var usage = rusage()
            getrusage(RUSAGE_SELF, &usage)
            let payload = try JSONEncoder().encode(result)
            let mapping = try JSONSerialization.jsonObject(with: payload)
            let data = try JSONSerialization.data(withJSONObject: ["outputs": mapping, "seconds": seconds,
                                                                    "host_peak_rss_bytes": usage.ru_maxrss])
            FileHandle.standardOutput.write(data + Data([10]))
        }
    }
}
