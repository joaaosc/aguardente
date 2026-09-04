import SwiftUI

/// Indicador de execução do centro da toolbar: progresso agregado e tempo decorrido.
///
/// Ocupa o lugar reservado pelo sistema a itens informativos (`.status`). Não repete
/// a fase nem a contagem de etapas — a fase vive no subtítulo da janela e o estado de
/// cada etapa já está nos ícones da barra lateral.
public struct RunActivity: View {
    let runner: PipelineRunner

    public init(runner: PipelineRunner) {
        self.runner = runner
    }

    public var body: some View {
        HStack(spacing: 10) {
            ProgressView(value: progress)
                .progressViewStyle(.linear)
                .frame(width: 130)

            Text(elapsedText)
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Progresso da execução")
        .accessibilityValue("\(Int(progress * 100)) por cento, \(elapsedText) decorridos")
    }

    /// Etapas concluídas mais a fração já cumprida da etapa em curso, quando ela reporta progresso.
    private var progress: Double {
        guard !runner.stages.isEmpty else { return 0 }
        let active = runner.stage(runner.activeStageID)
        let inFlight = active?.state == .running ? (active?.progress ?? 0) : 0
        return min((Double(runner.completedCount) + inFlight) / Double(runner.stages.count), 1)
    }

    private var elapsedText: String {
        let totalSeconds = Int(runner.elapsed.components.seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let seconds = totalSeconds % 60
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, seconds)
        }
        return String(format: "%02d:%02d", minutes, seconds)
    }
}
