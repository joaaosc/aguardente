import SwiftUI

public struct RootView: View {
    private let runner: PipelineRunner
    @State private var selection: Stage.ID?
    @State private var showReport = false
    @State private var isLogVisible = false

    public init(runner: PipelineRunner) {
        self.runner = runner
    }

    private var displayedStage: Stage? {
        runner.stage(selection ?? runner.activeStageID ?? runner.stages.first?.id)
    }

    public var body: some View {
        NavigationSplitView {
            StageList(runner: runner, selection: $selection)
                .navigationSplitViewColumnWidth(min: 200, ideal: 230, max: 320)
        } detail: {
            if let stage = displayedStage {
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

            ToolbarSpacer(.flexible)

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

            ToolbarSpacer(.fixed)

            ToolbarItem {
                Button {
                    isLogVisible.toggle()
                } label: {
                    Label("Log", systemImage: "square.bottomthird.inset.filled")
                }
                .help(isLogVisible ? "Oculta o log da etapa" : "Mostra o log da etapa")
            }

            ToolbarSpacer(.fixed)

            ToolbarItem {
                Menu {
                    Button("Relatório…", systemImage: "doc.text") { showReport = true }
                    Button("Revelar no Finder", systemImage: "folder") { runner.revealOutput() }
                    Divider()
                    Button("Limpar execução", systemImage: "arrow.counterclockwise", role: .destructive) {
                        runner.reset()
                    }
                } label: {
                    Label("Mais", systemImage: "ellipsis")
                }
            }
        }
        .sheet(isPresented: $showReport) {
            ReportSheet(runner: runner)
        }
    }
}
