import SwiftUI

public enum StageState: String, Codable, Sendable, Equatable, CaseIterable {
    case pending
    case running
    case ok
    case failed
    case skipped

    public var symbol: String {
        switch self {
        case .pending: "circle.dotted"
        case .running: "circle.dashed"
        case .ok:      "checkmark.circle.fill"
        case .failed:  "xmark.circle.fill"
        case .skipped: "minus.circle"
        }
    }

    /// Semantic color for stages.
    public var tint: Color {
        switch self {
        case .ok:      .green
        case .failed:  .red
        case .running: .accentColor
        case .pending, .skipped: .secondary
        }
    }

    public var labelDescription: String {
        switch self {
        case .pending: "Pendente"
        case .running: "Em execução"
        case .ok:      "Concluída com sucesso"
        case .failed:  "Falhou"
        case .skipped: "Ignorada"
        }
    }
}
