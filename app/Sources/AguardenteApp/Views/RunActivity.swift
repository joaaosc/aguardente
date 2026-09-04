import SwiftUI

/// Indicador de execução do centro da barra de ferramentas.
///
/// Ocupa o lugar que o sistema reserva a itens informativos (`.status`), e por
/// isso não desenha fundo próprio: a barra já é a camada de vidro, e empilhar
/// mais um material sobre ela é o que a HIG desaconselha ao falar de fundos
/// customizados em toolbar.
///
/// Não repete a fase nem o nome do modelo — os dois vivem no título da janela.
public struct RunActivity: View {
    let runner: PipelineRunner

    public init(runner: PipelineRunner) {
        self.runner = runner
    }

    public var body: some View {
        HStack(spacing: Espaco.interno) {
            ZStack {
                AnelDeProgresso(valor: progress, espessura: 2.5)
                    .frame(width: 18, height: 18)

                if runner.phase == .paused {
                    Image(systemName: "pause.fill")
                        .font(.system(size: 7))
                        .foregroundStyle(.secondary)
                }
            }

            Text("\(runner.completedCount)/\(runner.stages.count)")
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .contentTransition(.numericText())

            Divider()
                .frame(height: 12)

            Text(elapsedText)
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .contentTransition(.numericText())

            // O pipeline pode ficar sem emitir eventos por muito tempo numa
            // etapa longa. Sem este aviso, silêncio prolongado é
            // indistinguível de travamento.
            if runner.isStalled {
                Image(systemName: "clock.badge.exclamationmark")
                    .foregroundStyle(.orange)
                    .help("Sem novidades do pipeline há algum tempo")
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Progresso da execução")
        .accessibilityValue("\(Int(progress * 100)) por cento, "
                            + "\(runner.completedCount) de \(runner.stages.count) etapas, "
                            + "\(elapsedText) decorridos")
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
