import SwiftUI

public struct RootView: View {
    private let runner: PipelineRunner
    // A seleção e a visibilidade do log são estado de cena: o macOS espera que
    // a janela reabra como o usuário a deixou.
    @SceneStorage("etapaSelecionada") private var selection: Stage.ID?
    @SceneStorage("logVisivel") private var isLogVisible = false
    @State private var showReport = false
    @State private var confirmandoLimpeza = false
    @State private var motor: BinaryResolver.Resolution = BinaryResolver.resolve()

    public init(runner: PipelineRunner) {
        self.runner = runner
    }

    /// O CLI que o aplicativo executa está instalado?
    private var motorAusente: Bool {
        if case .notFound = motor { return true }
        return false
    }

    /// A etapa a exibir, ou `nil` quando a janela deve mostrar a visão geral.
    ///
    /// A visão geral é o destino de três situações distintas — nada
    /// selecionado, a linha de resumo selecionada, e uma seleção gravada que
    /// não corresponde mais a nenhuma etapa — e todas convergem aqui.
    private var displayedStage: Stage? {
        guard selection != StageList.visaoGeralID else { return nil }
        guard let selection else { return nil }
        return runner.stage(selection)
    }

    public var body: some View {
        NavigationSplitView {
            StageList(runner: runner, selection: $selection)
                .navigationSplitViewColumnWidth(min: 210, ideal: 240, max: 320)
        } detail: {
            detalhe
                .overlay(alignment: .bottomTrailing) { atalhoParaEtapaAtual }
        }
        .navigationTitle(runner.modelName)
        .navigationSubtitle(runner.phase.description)
        .toolbar { barraDeFerramentas }
        // Publica o estado da janela para a barra de menus. Ver
        // `ComandosDeVisualizacao`.
        .focusedSceneValue(\.logVisivel, $isLogVisible)
        .focusedSceneValue(\.relatorioVisivel, $showReport)
        .sheet(isPresented: $showReport) {
            ReportSheet(runner: runner)
        }
        // Descartar uma execução de horas é irreversível, e a ação fica a um
        // clique de distância no menu e a um atalho de distância no teclado.
        .confirmationDialog("Limpar a execução atual?",
                            isPresented: $confirmandoLimpeza,
                            titleVisibility: .visible) {
            Button("Limpar execução", role: .destructive) { runner.reset() }
            Button("Cancelar", role: .cancel) {}
        } message: {
            Text("O progresso exibido e os registros desta execução são descartados. "
                 + "Os arquivos já gravados em \(runner.outDir) permanecem no disco.")
        }
    }

    @ViewBuilder
    private var detalhe: some View {
        if motorAusente {
            motorNaoEncontrado
        } else if let stage = displayedStage {
            StageDetail(runner: runner,
                        stage: stage,
                        isLogVisible: $isLogVisible,
                        iniciar: { runner.start() })
        } else {
            RunOverview(runner: runner) { runner.start() }
        }
    }

    /// Atalho flutuante de volta para a etapa em curso.
    ///
    /// Numa execução de horas o usuário navega para trás para reler o log de
    /// uma etapa que já passou — e a lista não o traz de volta sozinha, porque
    /// sequestrar a navegação de quem escolheu manualmente seria pior. Este
    /// controle só existe enquanto há uma etapa em curso *e* ela não é a que
    /// está à vista, e some sozinho assim que deixa de ter função.
    ///
    /// É um controle real, flutuando sobre o conteúdo: o único lugar da janela
    /// onde Liquid Glass é o material certo.
    @ViewBuilder
    private var atalhoParaEtapaAtual: some View {
        if let ativa = runner.activeStageID,
           !motorAusente,
           displayedStage?.id != ativa,
           let etapa = runner.stage(ativa) {
            Button {
                withAnimation(.easeInOut(duration: 0.25)) { selection = ativa }
            } label: {
                HStack(spacing: Espaco.interno) {
                    Image(systemName: "arrow.down.circle.fill")
                    Text("Em execução: \(etapa.title)")
                        .lineLimit(1)
                }
                .font(.callout)
                .padding(.horizontal, Espaco.bloco - 2)
                .padding(.vertical, Espaco.interno + 2)
            }
            .buttonStyle(.plain)
            .vidro(em: Capsule())
            .padding(Espaco.bloco)
            .transition(.move(edge: .bottom).combined(with: .opacity))
            .help("Volta para a etapa que está sendo executada")
        }
    }

    /// Barra de ferramentas em dois grupos.
    ///
    /// A HIG limita a três grupos e pede uma única ação primária, na borda
    /// direita. O transporte da execução (parar, iniciar/pausar) é o grupo
    /// primário e fica por último; ver o log e as ações secundárias formam o
    /// grupo anterior.
    @ToolbarContentBuilder
    private var barraDeFerramentas: some ToolbarContent {
        ToolbarItem(placement: .status) {
            if runner.phase != .idle {
                RunActivity(runner: runner)
            }
        }

        espacador(.flexivel)

        ToolbarItemGroup {
            Button {
                isLogVisible.toggle()
            } label: {
                Label("Log", systemImage: "square.bottomthird.inset.filled")
            }
            .help(isLogVisible ? "Oculta o log da etapa" : "Mostra o log da etapa")

            Menu {
                Button("Relatório…", systemImage: "doc.text") { showReport = true }
                Button("Revelar no Finder", systemImage: "folder") { runner.revealOutput() }
                    .disabled(!runner.hasOutput)
                Divider()
                Button("Limpar execução", systemImage: "arrow.counterclockwise", role: .destructive) {
                    confirmandoLimpeza = true
                }
            } label: {
                Label("Mais", systemImage: "ellipsis")
            }
        }

        espacador(.fixo)

        ToolbarItem {
            Button {
                runner.cancel()
            } label: {
                Label("Parar", systemImage: "stop.fill")
            }
            .disabled(!runner.phase.isActive)
            .help("Encerra o pipeline em execução")
        }

        // Transporte da execução, em glifo simples.
        //
        // A tentativa anterior foi dar a este botão o estilo proeminente que a
        // HIG reserva à ação primária. Renderizado, o botão saiu idêntico ao
        // "Parar" ao lado: no macOS o SwiftUI ignora `buttonStyle` e `tint` em
        // item de barra de ferramentas — a proeminência ali vem de
        // `NSToolbarItem.style = .prominent`, que esta versão do SwiftUI não
        // expõe. Em vez de simular o efeito com desenho próprio, a ação
        // primária proeminente vive no conteúdo (`BotaoPrimario`), e a barra
        // fica com o glifo simples que Xcode e QuickTime usam para transporte.
        ToolbarItem {
            Button {
                runner.toggle()
            } label: {
                Label(
                    runner.isRunning ? "Pausar" : "Iniciar",
                    systemImage: runner.isRunning ? "pause.fill" : "play.fill"
                )
            }
            .help(runner.isRunning ? "Suspende o pipeline" : "Executa o pipeline")
        }
    }

    private enum TipoDeEspacador { case flexivel, fixo }

    /// Separador de barra de ferramentas, quando o sistema o oferece.
    ///
    /// `ToolbarSpacer` só existe a partir do macOS 26. Ele agrupa os botões em
    /// blocos visuais, e é a única razão pela qual a interface exigiria um
    /// sistema recente — o resto funciona desde o macOS 14. Ausente o
    /// separador, a barra continua completa, apenas sem os intervalos.
    @ToolbarContentBuilder
    private func espacador(_ tipo: TipoDeEspacador) -> some ToolbarContent {
        if #available(macOS 26.0, *) {
            switch tipo {
            case .flexivel: ToolbarSpacer(.flexible)
            case .fixo: ToolbarSpacer(.fixed)
            }
        }
    }

    /// Primeira barreira real de quem instala só o aplicativo.
    ///
    /// O `.app` não converte nada sozinho: ele executa o CLI, que por sua vez
    /// exige Python, torch e aria2. Sem essa tela o erro virava uma linha no
    /// log, que começa oculto — a janela apenas dizia "Falhou" sem explicar o
    /// que instalar.
    @ViewBuilder
    private var motorNaoEncontrado: some View {
        ContentUnavailableView {
            Label("Motor de conversão não encontrado", systemImage: "shippingbox")
        } description: {
            Text("O aplicativo executa o programa `aguardente`, que precisa estar "
                 + "instalado à parte junto com Python 3.12, torch e aria2.")
        } actions: {
            if case .notFound(let comando) = motor {
                Text(comando)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(Espaco.bloco - 4)
                    .background(Color(nsColor: .textBackgroundColor),
                                in: RoundedRectangle(cornerRadius: Raio.pequeno))
            }
            BotaoPrimario(titulo: "Verificar novamente", simbolo: "arrow.clockwise") {
                motor = BinaryResolver.resolve()
            }
        }
    }
}
