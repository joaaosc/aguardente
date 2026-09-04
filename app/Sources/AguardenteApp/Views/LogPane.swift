import SwiftUI
import AppKit

/// Painel inferior de log da etapa, com filtro e cópia na própria barra do painel.
///
/// Só as linhas de erro e aviso recebem cor: colorir todos os níveis faz o painel
/// competir com o resto da janela em vez de destacar o que exige atenção.
public struct LogPane: View {
    let lines: [LogLine]
    @State private var filterText: String = ""
    @State private var didCopy: Bool = false

    public init(lines: [LogLine]) {
        self.lines = lines
    }

    private var filteredLines: [LogLine] {
        guard !filterText.isEmpty else { return lines }
        return lines.filter { $0.text.localizedCaseInsensitiveContains(filterText) }
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            TextField("Filtrar", text: $filterText)
                .textFieldStyle(.roundedBorder)
                .controlSize(.small)
                .frame(maxWidth: 200)

            Spacer()

            Button {
                copyLog()
            } label: {
                Label(didCopy ? "Copiado" : "Copiar", systemImage: didCopy ? "checkmark" : "doc.on.doc")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
            .controlSize(.small)
            .disabled(lines.isEmpty)
            .help("Copia o log desta etapa")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.bar)
    }

    @ViewBuilder
    private var content: some View {
        if filteredLines.isEmpty {
            Text(lines.isEmpty ? "Sem registros nesta etapa." : "Nenhuma linha corresponde ao filtro.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .textBackgroundColor))
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(filteredLines) { line in
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
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
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                }
                .background(Color(nsColor: .textBackgroundColor))
                .textSelection(.enabled)
                .onChange(of: lines.count) { _, _ in irParaOFim(proxy) }
                // Filtrar reconstrói a lista inteira; sem isto a rolagem ficava
                // parada onde o log não filtrado estava, quase sempre fora da
                // faixa visível do resultado.
                .onChange(of: filterText) { _, _ in irParaOFim(proxy) }
            }
        }
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
