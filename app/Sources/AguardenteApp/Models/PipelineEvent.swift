import Foundation

public enum PipelineEvent: Sendable, Decodable {
    case plan(stages: [PlanStage])
    case stageStart(id: String, n: Int, of: Int, rationale: String)
    case stageEnd(id: String, ok: Bool, ms: Int)
    case progress(id: String, current: Int, total: Int?)
    case metric(id: String, key: String, value: Double, unit: String)
    case log(id: String, message: String, level: String)
    case error(id: String, message: String, hint: String?)
    case unknown(type: String)

    public struct PlanStage: Sendable, Decodable {
        public let id: String
        public let title: String
        public let rationale: String?

        public init(id: String, title: String, rationale: String? = nil) {
            self.id = id
            self.title = title
            self.rationale = rationale
        }
    }

    private enum CodingKeys: String, CodingKey {
        case ts, ev, stages, id, n, of, rationale, ok, ms, cur, tot, k, v, unit, msg, level, hint
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let ev = try container.decode(String.self, forKey: .ev)

        switch ev {
        case "plan":
            let stages = try container.decodeIfPresent([PlanStage].self, forKey: .stages) ?? []
            self = .plan(stages: stages)

        case "stage.start":
            let id = try container.decode(String.self, forKey: .id)
            let n = try container.decodeIfPresent(Int.self, forKey: .n) ?? 0
            let of = try container.decodeIfPresent(Int.self, forKey: .of) ?? 0
            let rationale = try container.decodeIfPresent(String.self, forKey: .rationale) ?? ""
            self = .stageStart(id: id, n: n, of: of, rationale: rationale)

        case "stage.end":
            let id = try container.decode(String.self, forKey: .id)
            let ok = try container.decode(Bool.self, forKey: .ok)
            let ms = try container.decodeIfPresent(Int.self, forKey: .ms) ?? 0
            self = .stageEnd(id: id, ok: ok, ms: ms)

        case "progress":
            let id = try container.decode(String.self, forKey: .id)
            let cur = try container.decode(Int.self, forKey: .cur)
            let tot = try container.decodeIfPresent(Int.self, forKey: .tot)
            self = .progress(id: id, current: cur, total: tot)

        case "metric":
            let id = try container.decode(String.self, forKey: .id)
            let k = try container.decode(String.self, forKey: .k)
            let v = try container.decode(Double.self, forKey: .v)
            let unit = try container.decodeIfPresent(String.self, forKey: .unit) ?? ""
            self = .metric(id: id, key: k, value: v, unit: unit)

        case "log":
            let id = try container.decodeIfPresent(String.self, forKey: .id) ?? "pipeline"
            let msg = try container.decode(String.self, forKey: .msg)
            let level = try container.decodeIfPresent(String.self, forKey: .level) ?? "info"
            self = .log(id: id, message: msg, level: level)

        case "error":
            let id = try container.decodeIfPresent(String.self, forKey: .id) ?? "pipeline"
            let msg = try container.decode(String.self, forKey: .msg)
            let hint = try container.decodeIfPresent(String.self, forKey: .hint)
            self = .error(id: id, message: msg, hint: hint)

        default:
            self = .unknown(type: ev)
        }
    }
}
