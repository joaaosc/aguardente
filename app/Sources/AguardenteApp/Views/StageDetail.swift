import SwiftUI

public struct StageDetail: View {
    let runner: PipelineRunner
    let stage: Stage
    @Binding var isLogVisible: Bool
    let iniciar: () -> Void

    public init(runner: PipelineRunner,
                stage: Stage,
                isLogVisible: Binding<Bool>,
                iniciar: @escaping () -> Void) {
        self.runner = runner
        self.stage = stage
        self._isLogVisible = isLogVisible
        self.iniciar = iniciar
    }

    private var indice: Int {
        (runner.stages.firstIndex { $0.id == stage.id } ?? 0) + 1
    }

    public var body: some View {
        VSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    cabecalho
                    corpo
                }
            }
            .bordaDeRolagem()
            .frame(minHeight: 220)

            if isLogVisible {
                LogPane(lines: stage.log)
                    .frame(minHeight: 140, idealHeight: 230)
            }
        }
        .background(.background)
    }

    /// Cabeçalho da etapa, estendido por baixo da barra lateral.
    ///
    /// Este bloco é a razão pela qual a janela deixou de parecer um formulário.
    /// A HIG pede que conteúdo visualmente rico corra por baixo da barra
    /// lateral em vez de parar numa borda dura — e é justamente essa
    /// continuidade que separa a camada de controle da de conteúdo. O
    /// gradiente é fraco de propósito: ele existe para dar lugar ao título,
    /// não para ser notado.
    private var cabecalho: some View {
        VStack(alignment: .leading, spacing: Espaco.bloco) {
            VStack(alignment: .leading, spacing: Espaco.interno) {
                HStack(spacing: Espaco.interno) {
                    Text("Etapa \(indice) de \(runner.stages.count)")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)

                    EstadoPill(state: stage.state)
                }

                Text(stage.title)
                    .font(.largeTitle.weight(.semibold))
                    .lineLimit(2)
                    .minimumScaleFactor(0.8)

                if !stage.rationale.isEmpty {
                    Text(stage.rationale)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            HStack(alignment: .bottom, spacing: Espaco.bloco) {
                StageTrack(stages: runner.stages, activeID: runner.activeStageID)
                    .frame(maxWidth: 380)

                Spacer(minLength: 0)

                // A ação primária precisa existir em toda tela, não só na visão
                // geral: quem abre o aplicativo numa etapa qualquer também
                // precisa de um lugar óbvio para começar.
                if runner.phase == .idle {
                    BotaoPrimario(titulo: "Iniciar", simbolo: "play.fill", acao: iniciar)
                }
            }
        }
        .padding(.horizontal, Espaco.secao)
        .padding(.top, Espaco.bloco)
        .padding(.bottom, Espaco.secao)
        .frame(maxWidth: 660, alignment: .leading)
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

    private var corpo: some View {
        VStack(alignment: .leading, spacing: Espaco.secao) {
            if stage.state == .running, let progress = stage.progress {
                // Rótulo e valor na mesma linha, acima da barra. O
                // `currentValueLabel` do `ProgressView` cai numa terceira
                // linha à esquerda, e o percentual acaba longe da ponta da
                // barra a que ele se refere.
                VStack(alignment: .leading, spacing: Espaco.minimo) {
                    HStack {
                        Text("Progresso da etapa")
                            .font(.subheadline)
                        Spacer()
                        Text(progress, format: .percent.precision(.fractionLength(0)))
                            .font(.subheadline.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .contentTransition(.numericText())
                    }

                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                        .labelsHidden()
                }
                .accessibilityElement(children: .combine)
            }

            if stage.state == .failed {
                Aviso(
                    texto: "Esta etapa falhou. O motivo está no log.",
                    simbolo: "exclamationmark.triangle.fill",
                    cor: .red
                )
            }

            // A conversão Core AI é a única etapa que exige um sistema mais
            // novo que o mínimo do aplicativo. Dizer isso aqui, antes de
            // executar, evita horas de destilação seguidas de uma falha na
            // última etapa.
            if stage.id == CoreAIDisponibilidade.etapaDependente,
               !CoreAIDisponibilidade.suportada {
                Aviso(
                    texto: CoreAIDisponibilidade.motivo,
                    simbolo: "exclamationmark.circle.fill",
                    cor: .orange
                )
            }

            if !stage.metrics.isEmpty {
                StageMetrics(metrics: stage.metrics)
            }

            if stage.metrics.isEmpty, stage.state == .pending {
                Text("Esta etapa ainda não começou. As métricas aparecem aqui "
                     + "conforme ela reporta resultados.")
                    .font(.callout)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // Com o log oculto, o corpo de uma etapa em curso fica vazio por
            // baixo das métricas — e o log é justamente o que se procura
            // enquanto se espera. Oferecê-lo aqui evita ter que descobrir o
            // botão da barra de ferramentas.
            if !isLogVisible, !stage.log.isEmpty {
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) { isLogVisible = true }
                } label: {
                    Label("Mostrar log desta etapa (\(stage.log.count) linhas)",
                          systemImage: "square.bottomthird.inset.filled")
                }
                .buttonStyle(.link)
                .font(.callout)
            }
        }
        .padding(.horizontal, Espaco.secao)
        .padding(.bottom, Espaco.secao)
        .frame(maxWidth: 660, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Selo de estado da etapa.
///
/// Fica na camada de conteúdo, e por isso usa material padrão em vez de
/// Liquid Glass: vidro em elemento de conteúdo confunde a hierarquia entre o
/// que é controle e o que é informação.
public struct EstadoPill: View {
    let state: StageState

    public init(state: StageState) {
        self.state = state
    }

    public var body: some View {
        HStack(spacing: Espaco.minimo) {
            Image(systemName: state.symbol)
            Text(state.labelDescription)
        }
        .font(.caption.weight(.medium))
        .foregroundStyle(state.tint)
        .padding(.horizontal, Espaco.interno)
        .padding(.vertical, 3)
        .background(state.tint.opacity(0.14), in: Capsule())
        .accessibilityElement(children: .combine)
        .accessibilityLabel(state.labelDescription)
    }
}

/// Caixa de aviso com símbolo, cor semântica e texto selecionável.
public struct Aviso: View {
    let texto: String
    let simbolo: String
    let cor: Color

    public init(texto: String, simbolo: String, cor: Color) {
        self.texto = texto
        self.simbolo = simbolo
        self.cor = cor
    }

    public var body: some View {
        Label {
            Text(texto)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: simbolo)
                .foregroundStyle(cor)
        }
        .font(.callout)
        .padding(Espaco.bloco - 4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(cor.opacity(0.10), in: RoundedRectangle(cornerRadius: Raio.medio))
        .overlay {
            RoundedRectangle(cornerRadius: Raio.medio)
                .strokeBorder(cor.opacity(0.30))
        }
    }
}
