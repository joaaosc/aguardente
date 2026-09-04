import SwiftUI
import AppKit

/// Painel inferior de log da etapa, com filtro, recorte por severidade e cópia
/// na própria barra do painel.
///
/// Só as linhas de erro e aviso recebem cor: colorir todos os níveis faz o
/// painel competir com o resto da janela em vez de destacar o que exige
/// atenção.
public struct LogPane: View {
    let lines: [LogLine]
    @State private var filterText: String = ""
    @State private var soProblemas: Bool = false
    @State private var didCopy: Bool = false

    public init(lines: [LogLine]) {
        self.lines = lines
    }

    private var filteredLines: [LogLine] {
        lines.filter { linha in
            if soProblemas && linha.levelSymbol == nil { return false }
            if filterText.isEmpty { return true }
            return linha.text.localizedCaseInsensitiveContains(filterText)
        }
    }

    /// Quantos problemas existem no log inteiro, não no recorte visível.
    ///
    /// O número precisa continuar verdadeiro depois de o usuário filtrar por
    /// texto — é ele que justifica ligar o recorte por severidade.
    private var totalDeProblemas: Int {
        lines.count { $0.levelSymbol != nil }
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
        }
    }

    private var header: some View {
        HStack(spacing: Espaco.interno) {
            TextField("Filtrar", text: $filterText)
                .textFieldStyle(.roundedBorder)
                .controlSize(.small)
                .frame(maxWidth: 220)

            if totalDeProblemas > 0 {
                Toggle(isOn: $soProblemas) {
                    Label("\(totalDeProblemas)", systemImage: "exclamationmark.triangle.fill")
                        .font(.caption.monospacedDigit())
                }
                .toggleStyle(.button)
                .controlSize(.small)
                .help(soProblemas ? "Mostra todas as linhas"
                                  : "Mostra só erros e avisos")
                .accessibilityLabel("Mostrar só erros e avisos")
            }

            Spacer()

            Text("\(filteredLines.count) linha\(filteredLines.count == 1 ? "" : "s")")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.tertiary)
                .contentTransition(.numericText())

            Button {
                copyLog()
            } label: {
                Label(didCopy ? "Copiado" : "Copiar",
                      systemImage: didCopy ? "checkmark" : "doc.on.doc")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
            .controlSize(.small)
            .disabled(filteredLines.isEmpty)
            .help("Copia as linhas visíveis")
        }
        .padding(.horizontal, Espaco.bloco - 4)
        .padding(.vertical, Espaco.interno - 2)
        .background(.bar)
    }

    @ViewBuilder
    private var content: some View {
        if filteredLines.isEmpty {
            estadoVazio
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(filteredLines) { line in
                            HStack(alignment: .firstTextBaseline, spacing: Espaco.interno) {
                                Text(line.formattedTime)
                                    .foregroundStyle(.tertiary)

                                // Severidade transmitida só por cor é invisível
                                // para quem não a distingue; o símbolo carrega a
                                // mesma informação de forma independente.
                                if let simbolo = line.levelSymbol {
                                    Image(systemName: simbolo)
                                        .foregroundStyle(style(for: line.level))
                                        .accessibilityLabel(line.level.lowercased().hasPrefix("err")
                                                            ? "Erro" : "Aviso")
                                }

                                Text(line.text)
                                    .foregroundStyle(style(for: line.level))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .id(line.id)
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                    .padding(.horizontal, Espaco.bloco - 4)
                    .padding(.vertical, Espaco.interno)
                }
                .background(Color(nsColor: .textBackgroundColor))
                .textSelection(.enabled)
                .onChange(of: lines.count) { _, _ in irParaOFim(proxy) }
                // Filtrar reconstrói a lista inteira; sem isto a rolagem ficava
                // parada onde o log não filtrado estava, quase sempre fora da
                // faixa visível do resultado.
                .onChange(of: filterText) { _, _ in irParaOFim(proxy) }
                .onChange(of: soProblemas) { _, _ in irParaOFim(proxy) }
            }
        }
    }

    private var estadoVazio: some View {
        VStack(spacing: Espaco.interno) {
            Image(systemName: lines.isEmpty ? "text.alignleft" : "line.3.horizontal.decrease")
                .font(.title2)
                .foregroundStyle(.tertiary)

            Text(mensagemDeVazio)
                .font(.callout)
                .foregroundStyle(.secondary)

            if !lines.isEmpty {
                Button("Limpar filtros") {
                    filterText = ""
                    soProblemas = false
                }
                .buttonStyle(.link)
                .controlSize(.small)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .textBackgroundColor))
    }

    private var mensagemDeVazio: String {
        if lines.isEmpty { return "Sem registros nesta etapa." }
        if soProblemas && filterText.isEmpty { return "Nenhum erro ou aviso nesta etapa." }
        return "Nenhuma linha corresponde ao filtro."
    }

    private func irParaOFim(_ proxy: ScrollViewProxy) {
        guard let ultima = filteredLines.last else { return }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo(ultima.id, anchor: .bottom)
        }
    }

    private func style(for level: String) -> Color {
        switch level.lowercased() {
        case "error", "erro": .red
        case "warn", "warning", "aviso": .orange
        default: .primary
        }
    }

    /// Copia o que está à vista.
    ///
    /// O botão fica ao lado do campo de filtro, e copiar o log inteiro quando a
    /// tela mostra três linhas contraria o que a posição promete.
    private func copyLog() {
        let text = filteredLines.map { "[\($0.formattedTime)] \($0.text)" }.joined(separator: "\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        didCopy = true
        Task {
            try? await Task.sleep(for: .seconds(2))
            didCopy = false
        }
    }
}
