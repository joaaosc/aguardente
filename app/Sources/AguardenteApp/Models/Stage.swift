import Foundation
import SwiftUI

@Observable @MainActor
public final class Stage: Identifiable {
    public let id: String
    public var title: String
    public var rationale: String
    public var state: StageState
    public var progress: Double?
    public var metrics: [Metric]
    public var log: [LogLine]
    public var duration: Duration?

    public init(
        id: String,
        title: String,
        rationale: String = "",
        state: StageState = .pending,
        progress: Double? = nil,
        metrics: [Metric] = [],
        log: [LogLine] = [],
        duration: Duration? = nil
    ) {
        self.id = id
        self.title = title
        self.rationale = rationale
        self.state = state
        self.progress = progress
        self.metrics = metrics
        self.log = log
        self.duration = duration
    }

    public func appendCappedLog(_ line: LogLine, limit: Int = 5_000) {
        log.append(line)
        if log.count > limit {
            log.removeFirst(log.count - limit)
        }
    }
}
