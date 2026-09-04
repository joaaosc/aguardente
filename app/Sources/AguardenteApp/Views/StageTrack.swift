import SwiftUI

/// Trilha do pipeline: um segmento por etapa, lado a lado.
///
/// Uma `ProgressView` linear diz que 40% do trabalho acabou, e nada além
/// disso. Este pipeline tem cinco etapas de custos muito diferentes — o
/// download leva minutos, a recuperação leva horas — e o que o usuário
/// pergunta ao olhar para a janela não é "quantos por cento" e sim "em que
/// ponto da sequência eu estou, e o que já passou deu certo?". Um segmento
/// por etapa responde às duas perguntas de uma vez; a barra única não
/// responde a nenhuma.
public struct StageTrack: View {
    private let stages: [Stage]
    private let activeID: Stage.ID?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// Alternado uma única vez no `onAppear`; a animação em `repeatForever`
    /// faz o resto sem timer nem redesenho por quadro.
    @State private var respirando = false

    public init(stages: [Stage], activeID: Stage.ID?) {
        self.stages = stages
        self.activeID = activeID
    }

    public var body: some View {
        HStack(spacing: 3) {
            ForEach(stages) { stage in
                segmento(for: stage)
            }
        }
        .frame(height: 6)
        .onAppear { respirando = true }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Progresso do pipeline")
        .accessibilityValue(resumoAcessivel)
    }

    @ViewBuilder
    private func segmento(for stage: Stage) -> some View {
        let preenchimento = fracao(of: stage)

        Capsule()
            .fill(.quaternary)
            .overlay(alignment: .leading) {
                GeometryReader { geo in
                    Capsule()
                        .fill(cor(for: stage))
                        .frame(width: geo.size.width * preenchimento)
                        .animation(.easeOut(duration: 0.35), value: preenchimento)
                }
            }
            .clipShape(Capsule())
            // A etapa em curso é a única que respira. Sem isso, uma janela
            // parada por quarenta minutos numa etapa longa fica visualmente
            // idêntica a uma janela travada.
            .opacity(deveRespirar(stage) && respirando ? 0.55 : 1)
            .animation(
                deveRespirar(stage)
                    ? .easeInOut(duration: 1.1).repeatForever(autoreverses: true)
                    : .default,
                value: respirando
            )
    }

    /// Quanto do segmento está preenchido.
    ///
    /// Etapa terminada preenche inteira, independentemente de ter reportado
    /// progresso. Etapa em curso preenche pelo progresso que ela informa — e,
    /// quando não informa nenhum, por uma fatia fixa que apenas sinaliza
    /// "esta é a atual", sem prometer uma medida que não existe.
    private func fracao(of stage: Stage) -> Double {
        switch stage.state {
        case .ok, .failed, .skipped: 1
        case .running: max(stage.progress ?? 0, 0.08)
        case .pending: 0
        }
    }

    private func cor(for stage: Stage) -> Color {
        switch stage.state {
        case .ok: .green
        case .failed: .red
        case .skipped: .secondary
        case .running: .accentColor
        case .pending: .clear
        }
    }

    private func deveRespirar(_ stage: Stage) -> Bool {
        stage.id == activeID && stage.state == .running && !reduceMotion
    }

    private var resumoAcessivel: String {
        let concluidas = stages.filter { $0.state == .ok }.count
        return "\(concluidas) de \(stages.count) etapas concluídas"
    }
}

#Preview("Trilha em execução") {
    StageTrack(
        stages: [
            Stage(id: "a", title: "Download", state: .ok),
            Stage(id: "b", title: "Poda", state: .ok),
            Stage(id: "c", title: "Logits", state: .running, progress: 0.42),
            Stage(id: "d", title: "Recuperação"),
            Stage(id: "e", title: "Core AI")
        ],
        activeID: "c"
    )
    .padding(40)
    .frame(width: 460)
}
