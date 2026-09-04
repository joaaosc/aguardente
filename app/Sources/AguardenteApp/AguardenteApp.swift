import SwiftUI

@main
struct AguardenteApp: App {
    @State private var runner = PipelineRunner()

    var body: some Scene {
        WindowGroup {
            RootView(runner: runner)
                // O detalhe passou a ter cabeçalho, trilha do pipeline e uma
                // grade de métricas. Abaixo desta largura a grade colapsa para
                // uma coluna e o cabeçalho perde a folga que o torna legível.
                .frame(minWidth: 840, minHeight: 560)
        }
        .windowStyle(.automatic)
        .windowToolbarStyle(.unified)
        .defaultSize(width: 1080, height: 720)
        .windowResizability(.contentMinSize)
        .commands {
            SidebarCommands()
            CommandGroup(replacing: .newItem) {}
            ComandosDeVisualizacao()

            CommandMenu("Execução") {
                Button(runner.isRunning ? "Pausar" : "Iniciar") {
                    runner.toggle()
                }
                .keyboardShortcut("r", modifiers: .command)

                Button("Parar") {
                    runner.cancel()
                }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(!runner.phase.isActive)

                Divider()

                // Ações que antes só existiam na barra de ferramentas. A regra
                // do macOS é que nada importante fique fora da barra de menus.
                Button("Revelar destino no Finder") {
                    runner.revealOutput()
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .disabled(!runner.hasOutput)

                Button("Limpar execução") {
                    runner.reset()
                }
                .keyboardShortcut(.delete, modifiers: [.command, .shift])

                Divider()

                Button("Carregar dados de demonstração") {
                    runner.loadDemoStages()
                }
            }
        }

        // A janela padrão de ⌘, é onde o macOS espera encontrar as
        // preferências. É também o único lugar em que o modelo, o destino e o
        // alvo podem ser escolhidos — antes eram constantes no código.
        Settings {
            SettingsView(runner: runner)
        }
    }
}
