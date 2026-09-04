import Foundation

public struct LogLine: Identifiable, Sendable, Equatable {
    public let id: UUID
    public let timestamp: Date
    public let text: String
    public let level: String

    /// Horário já formatado.
    ///
    /// Formatar sob demanda custava um `DateFormatter` novo por chamada, e o
    /// painel reavalia até 5.000 linhas a cada linha recebida: uma etapa
    /// falante pagava milhares de construções por redesenho. O custo aqui é
    /// pago uma vez, quando a linha nasce.
    public let formattedTime: String

    private static let horario: Date.FormatStyle = Date.FormatStyle()
        .hour(.twoDigits(amPM: .omitted))
        .minute(.twoDigits)
        .second(.twoDigits)

    public init(id: UUID = UUID(), timestamp: Date = Date(), text: String, level: String = "info") {
        self.id = id
        self.timestamp = timestamp
        self.text = text
        self.level = level
        self.formattedTime = timestamp.formatted(LogLine.horario)
    }

    /// Símbolo do nível, para que a severidade não dependa só da cor.
    public var levelSymbol: String? {
        switch level.lowercased() {
        case "error", "erro": "exclamationmark.octagon.fill"
        case "warn", "warning", "aviso": "exclamationmark.triangle.fill"
        default: nil
        }
    }
}
