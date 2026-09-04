import SwiftUI

public struct StageList: View {
    let runner: PipelineRunner
    @Binding var selection: Stage.ID?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Identificador da linha de resumo, que não corresponde a nenhuma etapa.
    ///
    /// A seleção da barra lateral é gravada em `SceneStorage` e já é um
    /// `Stage.ID` (uma `String`). Reservar um valor em vez de trocar o tipo por
    /// um enum mantém a janela reabrindo onde o usuário a deixou, inclusive
    /// para as cenas já gravadas antes desta linha existir.
    public static let visaoGeralID = "__visao-geral__"

    public init(runner: PipelineRunner, selection: Binding<Stage.ID?>) {
        self.runner = runner
        self._selection = selection
    }

    public var body: some View {
        ScrollViewReader { proxy in
            List(selection: $selection) {
                Section {
                    linhaDeVisaoGeral
                }

                Section("Pipeline") {
                    ForEach(Array(runner.stages.enumerated()), id: \.element.id) { indice, stage in
                        linha(for: stage, numero: indice + 1)
                            .id(stage.id)
                            .tag(stage.id)
                    }
                }
            }
            .listStyle(.sidebar)
            .onChange(of: runner.activeStageID) { _, new in
                guard let new, selection == nil || selection == runner.previousStageID else {
                    return // Não sequestra a navegação se o usuário escolheu outra manualmente
                }
                withAnimation(.easeInOut(duration: 0.3)) {
                    proxy.scrollTo(new, anchor: .center)
                }
            }
        }
    }

    /// Ponto de partida da janela: o estado da execução como um todo.
    ///
    /// Sem esta linha, abrir o aplicativo caía numa etapa arbitrária ou num
    /// aviso de "nenhuma etapa selecionada" — nos dois casos a primeira tela
    /// não dizia nada sobre o que estava prestes a acontecer.
    private var linhaDeVisaoGeral: some View {
        Label {
            Text("Visão geral")
        } icon: {
            Image(systemName: "chart.bar.doc.horizontal")
                .foregroundStyle(.tint)
        }
        .tag(Self.visaoGeralID)
        .accessibilityLabel("Visão geral da execução")
    }

    private func linha(for stage: Stage, numero: Int) -> some View {
        HStack(spacing: Espaco.interno) {
            Image(systemName: stage.state.symbol)
                .foregroundStyle(stage.state.tint)
                .symbolEffect(.pulse, isActive: stage.state == .running && !reduceMotion)
                // Sem largura fixa os glifos têm larguras diferentes e os
                // títulos ficam desalinhados de uma linha para a outra.
                .frame(width: 16)

            Text(stage.title)
                .foregroundStyle(stage.state == .pending ? .secondary : .primary)
                .lineLimit(1)

            Spacer(minLength: Espaco.minimo)

            marcador(for: stage, numero: numero)
        }
        .padding(.vertical, 3)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Etapa \(numero), \(stage.title), \(stage.state.labelDescription)"
                            + (stage.duration.map { ", \(formatDuration($0))" } ?? ""))
    }

    /// O que aparece à direita muda com o estado, porque a pergunta muda.
    ///
    /// Numa etapa concluída interessa quanto tempo ela custou; na etapa em
    /// curso interessa o quanto falta; numa etapa que ainda não começou não há
    /// nada a dizer além da posição dela na fila.
    @ViewBuilder
    private func marcador(for stage: Stage, numero: Int) -> some View {
        switch stage.state {
        case .running:
            if let progresso = stage.progress {
                Text(progresso, format: .percent.precision(.fractionLength(0)))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .contentTransition(.numericText())
            } else {
                ProgressView()
                    .controlSize(.mini)
            }

        case .ok, .failed, .skipped:
            if let duracao = stage.duration {
                Text(formatDuration(duracao))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

        case .pending:
            Text("\(numero)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
    }

    private func formatDuration(_ duration: Duration) -> String {
        let totalSeconds = Int(duration.components.seconds)
        let minutes = totalSeconds / 60
        let seconds = totalSeconds % 60
        if minutes > 0 {
            return String(format: "%02d:%02d", minutes, seconds)
        }
        return String(format: "%ds", seconds)
    }
}
