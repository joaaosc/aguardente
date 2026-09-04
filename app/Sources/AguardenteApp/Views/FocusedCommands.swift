import SwiftUI

/// Ponte entre o estado da janela e a barra de menus.
///
/// A HIG é explícita: como a barra de ferramentas pode ser customizada ou
/// ocultada, **todo** item dela precisa existir também como comando de menu.
/// Mostrar o log e abrir o relatório viviam só na barra — quem escondesse a
/// barra perdia as duas ações.
///
/// O estado das duas é da cena, não do modelo, então a barra de menus o
/// alcança por `focusedSceneValue`: a janela em foco publica os bindings, e os
/// comandos leem de quem estiver na frente.
struct LogVisivelKey: FocusedValueKey {
    typealias Value = Binding<Bool>
}

struct RelatorioVisivelKey: FocusedValueKey {
    typealias Value = Binding<Bool>
}

extension FocusedValues {
    var logVisivel: Binding<Bool>? {
        get { self[LogVisivelKey.self] }
        set { self[LogVisivelKey.self] = newValue }
    }

    var relatorioVisivel: Binding<Bool>? {
        get { self[RelatorioVisivelKey.self] }
        set { self[RelatorioVisivelKey.self] = newValue }
    }
}

/// Comandos do menu Visualizar que espelham a barra de ferramentas.
struct ComandosDeVisualizacao: Commands {
    @FocusedBinding(\.logVisivel) private var logVisivel
    @FocusedBinding(\.relatorioVisivel) private var relatorioVisivel

    var body: some Commands {
        CommandGroup(after: .sidebar) {
            Button(logVisivel == true ? "Ocultar Log" : "Mostrar Log") {
                logVisivel?.toggle()
            }
            .keyboardShortcut("l", modifiers: [.command, .shift])
            .disabled(logVisivel == nil)

            Divider()

            Button("Relatório da Execução…") {
                relatorioVisivel = true
            }
            .keyboardShortcut("i", modifiers: .command)
            .disabled(relatorioVisivel == nil)
        }
    }
}
