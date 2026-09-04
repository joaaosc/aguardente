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

    /// Acrescenta uma linha e descarta o excesso em blocos.
    ///
    /// Aparar uma linha por vez faria o array inteiro deslizar a cada linha
    /// nova depois de atingido o teto — cinco mil deslocamentos por linha numa
    /// etapa falante. Cortando uma folga de 10% de uma só vez, o próximo corte
    /// só acontece quinhentas linhas depois, e o custo por linha volta a ser
    /// constante na média.
    public func appendCappedLog(_ line: LogLine, limit: Int = 5_000) {
        log.append(line)
        guard log.count > limit else { return }
        let alvo = limit - limit / 10
        log.removeFirst(log.count - alvo)
    }
}
