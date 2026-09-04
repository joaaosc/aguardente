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

    private var displayedStage: Stage? {
        runner.stage(selection ?? runner.activeStageID ?? runner.stages.first?.id)
    }

    public var body: some View {
        NavigationSplitView {
            StageList(runner: runner, selection: $selection)
                .navigationSplitViewColumnWidth(min: 200, ideal: 230, max: 320)
        } detail: {
            if motorAusente {
                motorNaoEncontrado
            } else if let stage = displayedStage {
                StageDetail(stage: stage, isLogVisible: isLogVisible)
            } else {
                ContentUnavailableView(
                    "Nenhuma etapa",
                    systemImage: "sidebar.left",
                    description: Text("Selecione uma etapa na barra lateral.")
                )
            }
        }
        .navigationTitle(runner.modelName)
        .navigationSubtitle(runner.phase.description)
        .toolbar {
            ToolbarItem(placement: .status) {
                if runner.phase != .idle {
                    RunActivity(runner: runner)
                }
            }

            espacador(.flexivel)

            ToolbarItemGroup {
                Button {
                    runner.toggle()
                } label: {
                    Label(
                        runner.isRunning ? "Pausar" : "Iniciar",
                        systemImage: runner.isRunning ? "pause.fill" : "play.fill"
                    )
                }
                .help(runner.isRunning ? "Suspende o pipeline" : "Executa o pipeline")

                Button {
                    runner.cancel()
                } label: {
                    Label("Parar", systemImage: "stop.fill")
                }
                .disabled(!runner.phase.isActive)
                .help("Encerra o pipeline em execução")
            }

            espacador(.fixo)

            ToolbarItem {
                Button {
                    isLogVisible.toggle()
                } label: {
                    Label("Log", systemImage: "square.bottomthird.inset.filled")
                }
                .help(isLogVisible ? "Oculta o log da etapa" : "Mostra o log da etapa")
            }

            espacador(.fixo)

            ToolbarItem {
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
        }
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

    private enum Espaco { case flexivel, fixo }

    /// Separador de barra de ferramentas, quando o sistema o oferece.
    ///
    /// `ToolbarSpacer` só existe a partir do macOS 26. Ele agrupa os botões em
    /// blocos visuais, e é a única razão pela qual a interface exigiria um
    /// sistema recente — o resto funciona desde o macOS 14. Ausente o
    /// separador, a barra continua completa, apenas sem os intervalos.
    @ToolbarContentBuilder
    private func espacador(_ tipo: Espaco) -> some ToolbarContent {
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
                    .padding(10)
                    .background(Color(nsColor: .textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            Button("Verificar novamente") { motor = BinaryResolver.resolve() }
                .buttonStyle(.borderedProminent)
        }
    }
}
