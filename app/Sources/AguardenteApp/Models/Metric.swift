import Foundation

public struct Metric: Identifiable, Sendable, Equatable, Hashable {
    public let key: String
    public let label: String
    public let value: Double
    public let unit: String
    public let meetsTarget: Bool

    /// A chave da métrica é a identidade.
    ///
    /// Com um `UUID` novo a cada construção, atualizar o valor de uma métrica
    /// produzia um elemento de identidade diferente, e o SwiftUI tratava a
    /// atualização como remoção mais inserção — a linha reanimava do zero a
    /// cada número recebido. A chave é estável e já é única dentro da etapa.
    public var id: String { key }

    public init(key: String, label: String? = nil, value: Double, unit: String = "", meetsTarget: Bool = true) {
        self.key = key
        self.label = label ?? Metric.defaultLabel(for: key)
        self.value = value
        self.unit = unit
        self.meetsTarget = meetsTarget
    }

    public var formatted: String {
        if !unit.isEmpty {
            if value >= 1_000_000_000 {
                return String(format: "%.2f G%@", value / 1_000_000_000, unit)
            } else if value >= 1_000_000 {
                return String(format: "%.2f M%@", value / 1_000_000, unit)
            } else if value >= 1_000 {
                return String(format: "%.1f k%@", value / 1_000, unit)
            }
            return String(format: "%.2f %@", value, unit)
        }

        if key.contains("ppl") || key.contains("loss") {
            return String(format: "%.2f", value)
        }
        if key.contains("fraction") || key.contains("ratio") {
            return String(format: "%.1f%%", value * 100)
        }
        if value.truncatingRemainder(dividingBy: 1) == 0 {
            return String(format: "%.0f", value)
        }
        return String(format: "%.2f", value)
    }

    private static func defaultLabel(for key: String) -> String {
        switch key {
        case "ppl_teacher": return "Perplexidade Original"
        case "ppl_pruned": return "Pós-Poda"
        case "ppl_recovered": return "Perplexidade Final"
        case "recovered_fraction": return "Queda Recuperada"
        case "bundle_bytes": return "Tamanho do Bundle"
        case "params_target": return "Alvo de Parâmetros"
        case "seconds": return "Duração"
        default:
            return key.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
}
