import SwiftUI

@main
struct AguardenteApp: App {
    @State private var runner = PipelineRunner()

    var body: some Scene {
        WindowGroup {
            RootView(runner: runner)
                .frame(minWidth: 860, minHeight: 540)
        }
        .windowStyle(.automatic)
        .windowToolbarStyle(.unified)
        .commands {
            SidebarCommands()
            CommandGroup(replacing: .newItem) {}

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
    }
}
