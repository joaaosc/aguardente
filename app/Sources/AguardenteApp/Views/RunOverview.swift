import SwiftUI
import Charts

/// Tela inicial da janela: o estado da execução inteira.
///
/// O lugar que esta view ocupa era um `ContentUnavailableView` dizendo
/// "Selecione uma etapa na barra lateral" — ou seja, a primeira coisa que o
/// aplicativo mostrava era uma instrução para sair dela. Uma execução tem
/// estado próprio, anterior ao de qualquer etapa: o que está sendo destilado,
/// para onde, quanto já andou e quanto tempo levou. É isso que abre a janela
/// agora.
public struct RunOverview: View {
    let runner: PipelineRunner
    let iniciar: () -> Void

    public init(runner: PipelineRunner, iniciar: @escaping () -> Void) {
        self.runner = runner
        self.iniciar = iniciar
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                cabecalho
                corpo
            }
        }
        .bordaDeRolagem()
        .background(.background)
    }

    private var cabecalho: some View {
        VStack(alignment: .leading, spacing: Espaco.bloco) {
            VStack(alignment: .leading, spacing: Espaco.interno) {
                Text("Destilação")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)

                Text(runner.modelName)
                    .font(.largeTitle.weight(.semibold))
                    .lineLimit(2)
                    .minimumScaleFactor(0.7)
                    .textSelection(.enabled)

                Text(runner.phase.description)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            HStack(alignment: .center, spacing: Espaco.secao) {
                anel

                VStack(alignment: .leading, spacing: Espaco.interno) {
                    StageTrack(stages: runner.stages, activeID: runner.activeStageID)
                        .frame(maxWidth: 320)

                    Text("\(runner.completedCount) de \(runner.stages.count) etapas concluídas")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 0)

                if runner.phase == .idle {
                    BotaoPrimario(
                        titulo: "Iniciar destilação",
                        simbolo: "play.fill",
                        acao: iniciar
                    )
                }
            }
        }
        .padding(.horizontal, Espaco.secao)
        .padding(.top, Espaco.bloco)
        .padding(.bottom, Espaco.secao)
        .frame(maxWidth: 720, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(alignment: .bottom) {
            Marca.cabecalho
                // Onde o sistema estende o conteúdo por baixo da barra
                // lateral, o gradiente acompanha. Num `NavigationSplitView` de
                // macOS as duas colunas são adjacentes e não sobrepostas, então
                // hoje isto não muda nada aqui — fica pelo custo zero e porque
                // é a chamada correta para a intenção.
                .extensaoDeFundo()
                .overlay(alignment: .bottom) {
                    Rectangle()
                        .fill(.separator)
                        .frame(height: 1)
                }
        }
    }

    private var anel: some View {
        ZStack {
            AnelDeProgresso(valor: progresso, espessura: 5)

            VStack(spacing: 0) {
                Text(progresso, format: .percent.precision(.fractionLength(0)))
                    .font(.title3.weight(.semibold).monospacedDigit())
                    .contentTransition(.numericText())

                Text(tempoDecorrido)
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: 78, height: 78)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Progresso da execução")
        .accessibilityValue("\(Int(progresso * 100)) por cento, \(tempoDecorrido) decorridos")
    }

    private var corpo: some View {
        VStack(alignment: .leading, spacing: Espaco.secao) {
            configuracao

            if temDuracoes {
                linhaDoTempo
            } else {
                roteiro
            }
        }
        .padding(.horizontal, Espaco.secao)
        .padding(.bottom, Espaco.secao)
        .frame(maxWidth: 720, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var configuracao: some View {
        VStack(alignment: .leading, spacing: Espaco.interno) {
            Text("Configuração")
                .font(.headline)

            VStack(spacing: 0) {
                linhaDeConfiguracao("Modelo de origem", runner.modelName)
                Divider()
                linhaDeConfiguracao("Destino", runner.outDir)
                Divider()
                linhaDeConfiguracao("Tamanho do resultado", alvoDescrito)
            }
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: Raio.medio))
        }
    }

    /// Uma linha da tabela de configuração.
    ///
    /// `LabeledContent` seria a escolha idiomática, mas fora de um `Form` ele
    /// alinha o rótulo à direita, numa coluna calculada — na captura os três
    /// rótulos apareciam flutuando no meio da caixa, colados aos valores. Um
    /// `HStack` com `Spacer` produz o que a tabela promete: rótulo encostado à
    /// esquerda, valor à direita, e a coluna de valores lida por comparação.
    private func linhaDeConfiguracao(_ rotulo: String, _ valor: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: Espaco.bloco) {
            Text(rotulo)
                .font(.callout)

            Spacer(minLength: Espaco.bloco)

            Text(valor)
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.head)
                .textSelection(.enabled)
                .help(valor)
        }
        .padding(.horizontal, Espaco.bloco - 4)
        .padding(.vertical, Espaco.interno + 2)
        .accessibilityElement(children: .combine)
    }

    /// Quanto tempo cada etapa custou, na mesma escala.
    ///
    /// A barra lateral mostra as durações uma a uma, e ler cinco números
    /// isolados não diz onde o tempo foi gasto. Numa escala compartilhada, a
    /// etapa que domina a execução aparece sozinha — que é a pergunta que se
    /// faz depois de esperar horas.
    private var linhaDoTempo: some View {
        VStack(alignment: .leading, spacing: Espaco.interno) {
            Text("Tempo por etapa")
                .font(.headline)

            Chart {
                ForEach(runner.stages.filter { $0.duration != nil }) { stage in
                    BarMark(
                        x: .value("Segundos", segundos(stage.duration)),
                        y: .value("Etapa", stage.title)
                    )
                    .foregroundStyle(stage.state.tint.opacity(0.85))
                    .clipShape(RoundedRectangle(cornerRadius: 5))
                    .annotation(position: .trailing, alignment: .leading) {
                        Text(rotuloDeDuracao(stage.duration))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .chartXAxis(.hidden)
            .chartYAxis {
                AxisMarks(preset: .aligned, position: .leading) { _ in
                    AxisValueLabel()
                }
            }
            .frame(height: CGFloat(runner.stages.filter { $0.duration != nil }.count) * 34 + 16)
        }
    }

    /// O que vai acontecer, antes de acontecer.
    ///
    /// Numa execução que ainda não começou não há duração para plotar, e a
    /// metade de baixo da janela ficava vazia. O roteiro usa a explicação que
    /// cada etapa já carrega: quem abre o aplicativo pela primeira vez fica
    /// sabendo o que são poda, logits e recuperação sem precisar clicar em
    /// cinco linhas da barra lateral.
    private var roteiro: some View {
        VStack(alignment: .leading, spacing: Espaco.interno) {
            Text("O que será feito")
                .font(.headline)

            VStack(spacing: 0) {
                ForEach(Array(runner.stages.enumerated()), id: \.element.id) { indice, stage in
                    if indice > 0 { Divider() }

                    HStack(alignment: .top, spacing: Espaco.bloco - 4) {
                        Text("\(indice + 1)")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 16, alignment: .trailing)

                        VStack(alignment: .leading, spacing: Espaco.minimo) {
                            Text(stage.title)
                                .font(.callout.weight(.medium))

                            if !stage.rationale.isEmpty {
                                Text(stage.rationale)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }

                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, Espaco.bloco - 4)
                    .padding(.vertical, Espaco.interno + 2)
                    .accessibilityElement(children: .combine)
                }
            }
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: Raio.medio))
        }
    }

    private var temDuracoes: Bool {
        runner.stages.contains { $0.duration != nil }
    }

    private func segundos(_ duracao: Duration?) -> Double {
        guard let duracao else { return 0 }
        return Double(duracao.components.seconds)
    }

    private func rotuloDeDuracao(_ duracao: Duration?) -> String {
        let total = Int(segundos(duracao))
        let minutos = total / 60
        let segundos = total % 60
        return minutos > 0 ? String(format: "%d min %02d s", minutos, segundos)
                           : String(format: "%d s", segundos)
    }

    private var progresso: Double {
        guard !runner.stages.isEmpty else { return 0 }
        let ativa = runner.stage(runner.activeStageID)
        let emCurso = ativa?.state == .running ? (ativa?.progress ?? 0) : 0
        return min((Double(runner.completedCount) + emCurso) / Double(runner.stages.count), 1)
    }

    private var tempoDecorrido: String {
        let total = Int(runner.elapsed.components.seconds)
        let horas = total / 3600
        let minutos = (total % 3600) / 60
        let segundos = total % 60
        if horas > 0 {
            return String(format: "%d:%02d:%02d", horas, minutos, segundos)
        }
        return String(format: "%02d:%02d", minutos, segundos)
    }

    private var alvoDescrito: String {
        guard let alvo = runner.targetParams, alvo > 0 else {
            return "automático (pela RAM da máquina)"
        }
        return String(format: "%.2f B parâmetros", alvo / 1e9)
    }
}

/// Ação primária da tela, com o estilo que o sistema reserva a ela.
///
/// A HIG pede um único ponto focal por tela e manda tingir o **fundo** do
/// controle, não o símbolo nem o texto. `.glassProminent` faz exatamente isso
/// no macOS 26; antes dele, `.borderedProminent` é o equivalente da época.
public struct BotaoPrimario: View {
    let titulo: String
    let simbolo: String
    let acao: () -> Void

    public init(titulo: String, simbolo: String, acao: @escaping () -> Void) {
        self.titulo = titulo
        self.simbolo = simbolo
        self.acao = acao
    }

    public var body: some View {
        if #available(macOS 26.0, *) {
            botao.buttonStyle(.glassProminent)
        } else {
            botao.buttonStyle(.borderedProminent)
        }
    }

    private var botao: some View {
        Button(action: acao) {
            Label(titulo, systemImage: simbolo)
        }
        .controlSize(.large)
    }
}
