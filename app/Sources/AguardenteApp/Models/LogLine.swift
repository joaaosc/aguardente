import Foundation

public struct LogLine: Identifiable, Sendable, Equatable {
    public let id: UUID
    public let timestamp: Date
    public let text: String
    public let level: String

    public init(id: UUID = UUID(), timestamp: Date = Date(), text: String, level: String = "info") {
        self.id = id
        self.timestamp = timestamp
        self.text = text
        self.level = level
    }

    public var formattedTime: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: timestamp)
    }
}
