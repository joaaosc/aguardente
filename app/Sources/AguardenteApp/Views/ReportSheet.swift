import SwiftUI
import AppKit
import UniformTypeIdentifiers

public struct ReportSheet: View {
    let runner: PipelineRunner
    @Environment(\.dismiss) private var dismiss
    @State private var isCopied: Bool = false
    @State private var showFileExporter: Bool = false
    @State private var reportDocument: ReportDocument = ReportDocument(text: "")

    public init(runner: PipelineRunner) {
        self.runner = runner
    }

    /// Sem alvo informado quem dimensiona é o pipeline, pela RAM da máquina —
    /// e o relatório precisa dizer isso, não inventar um número.
    private var alvoDescrito: String {
        guard let alvo = runner.targetParams, alvo > 0 else {
            return "automático (dimensionado pela RAM)"
        }
        return String(format: "%.2f B", alvo / 1e9)
    }

    private var reportMarkdown: String {
        var md = "# Relatório de Execução · aguardente\n\n"
        md += "**Modelo:** `\(runner.modelName)`  \n"
        md += "**Destino:** `\(runner.outDir)`  \n"
        md += "**Alvo de Parâmetros:** \(alvoDescrito)  \n"
        md += "**Estado:** \(runner.phase.description)  \n"
        md += "**Tempo Total:** \(formatDuration(runner.elapsed))\n\n"

        md += "## Etapas do Pipeline\n\n"
        md += "| # | Etapa | Estado | Duração |\n"
        md += "|---|---|---|---|\n"
        for (idx, stage) in runner.stages.enumerated() {
            let dur = stage.duration.map { formatDuration($0) } ?? "—"
            md += "| \(idx + 1) | \(stage.title) | \(stage.state.labelDescription) | \(dur) |\n"
        }
        md += "\n"

        // A etapa entra na tabela porque a mesma chave pode ser emitida por
        // etapas diferentes — `seconds` vem de todas elas. Sem a coluna, duas
        // linhas idênticas apareceriam sem meio de distingui-las.
        if runner.stages.contains(where: { !$0.metrics.isEmpty }) {
            md += "## Métricas de Qualidade\n\n"
            md += "| Etapa | Métrica | Valor | Meta Atendida |\n"
            md += "|---|---|---|---|\n"
            for etapa in runner.stages {
                for m in etapa.metrics {
                    md += "| \(etapa.title) | \(m.label) | \(m.formatted) | \(m.meetsTarget ? "✓" : "⚠") |\n"
                }
            }
            md += "\n"
        }

        md += "## Como testar o modelo gerado\n\n"
        // Mesma fonte do bloco exibido na tela: dois literais divergem na
        // primeira vez que o comando muda.
        md += "```bash\n\(comandoDeTesteTexto)\n```\n"
        return md
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    cabecalho
                    corpo
                }
            }
            .bordaDeRolagem()
            .navigationTitle("Relatório da Execução")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Fechar") { dismiss() }
                }

                // Dois botões num mesmo `ToolbarItemGroup(placement:
                // .primaryAction)` numa folha de macOS renderizam só o
                // primeiro — o "Exportar…" simplesmente não aparecia na barra
                // inferior. Em itens separados, com o exportar em
                // `.confirmationAction`, os dois ficam visíveis e o principal
                // ganha a posição que o sistema reserva a ele.
                ToolbarItem(placement: .automatic) {
                    Button {
                        copyMarkdown()
                    } label: {
                        Label(isCopied ? "Copiado!" : "Copiar Markdown",
                              systemImage: isCopied ? "checkmark" : "doc.on.doc")
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        reportDocument = ReportDocument(text: reportMarkdown)
                        showFileExporter = true
                    } label: {
                        Label("Exportar…", systemImage: "square.and.arrow.up")
                    }
                }
            }
            .fileExporter(
                isPresented: $showFileExporter,
                document: reportDocument,
                contentType: .plainText,
                defaultFilename: "aguardente-relatorio.md"
            ) { _ in }
        }
        .frame(minWidth: 560, minHeight: 520)
    }

    /// Mesmo cabeçalho da janela principal, pelas mesmas razões.
    ///
    /// O relatório é a folha que o usuário exporta e mostra a outra pessoa; se
    /// ele tivesse tipografia e espaçamento próprios, seria uma segunda
    /// linguagem visual dentro do mesmo aplicativo.
    private var cabecalho: some View {
        VStack(alignment: .leading, spacing: Espaco.bloco) {
            VStack(alignment: .leading, spacing: Espaco.interno) {
                Text("Relatório")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)

                Text(runner.modelName)
                    .font(.title.weight(.semibold))
                    .lineLimit(2)
                    .minimumScaleFactor(0.7)
                    .textSelection(.enabled)

                Text("Poda estruturada, destilação e conversão Core AI")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: Espaco.secao) {
                fato("Estado", runner.phase.description)
                fato("Tempo total", formatDuration(runner.elapsed))
                fato("Alvo", alvoDescrito)
            }
        }
        .padding(.horizontal, Espaco.secao)
        .padding(.top, Espaco.bloco)
        .padding(.bottom, Espaco.secao)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(alignment: .bottom) {
            Marca.cabecalho
                .overlay(alignment: .bottom) {
                    Rectangle().fill(.separator).frame(height: 1)
                }
        }
    }

    private func fato(_ rotulo: String, _ valor: String) -> some View {
        VStack(alignment: .leading, spacing: Espaco.minimo) {
            Text(rotulo)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(valor)
                .font(.callout.weight(.medium).monospacedDigit())
                .lineLimit(1)
        }
        .accessibilityElement(children: .combine)
    }

    private var corpo: some View {
        VStack(alignment: .leading, spacing: Espaco.secao) {
            etapas

            // Métricas agrupadas por etapa. A identidade de `Metric` é a chave,
            // única dentro de uma etapa mas não entre elas: achatar tudo numa
            // lista só daria ids repetidos ao `ForEach`, que descartaria linhas
            // em silêncio.
            ForEach(runner.stages.filter { !$0.metrics.isEmpty }) { etapa in
                VStack(alignment: .leading, spacing: Espaco.interno) {
                    Text(etapa.title)
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.secondary)

                    StageMetrics(metrics: etapa.metrics, titulo: nil)
                }
            }

            comandoDeTeste
        }
        .padding(.horizontal, Espaco.secao)
        .padding(.bottom, Espaco.secao)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var etapas: some View {
        VStack(alignment: .leading, spacing: Espaco.interno) {
            Text("Etapas")
                .font(.headline)

            VStack(spacing: 0) {
                ForEach(Array(runner.stages.enumerated()), id: \.element.id) { indice, stage in
                    if indice > 0 { Divider() }

                    HStack(spacing: Espaco.bloco - 4) {
                        Text("\(indice + 1)")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.tertiary)
                            .frame(width: 16, alignment: .trailing)

                        Text(stage.title)
                            .font(.callout)

                        Spacer(minLength: Espaco.interno)

                        EstadoPill(state: stage.state)

                        Text(stage.duration.map { formatDuration($0) } ?? "—")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 92, alignment: .trailing)
                    }
                    .padding(.horizontal, Espaco.bloco - 4)
                    .padding(.vertical, Espaco.interno + 2)
                    .accessibilityElement(children: .combine)
                }
            }
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: Raio.medio))
        }
    }

    private var comandoDeTeste: some View {
        VStack(alignment: .leading, spacing: Espaco.interno) {
            Text("Como testar o modelo gerado")
                .font(.headline)

            Text(comandoDeTesteTexto)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .padding(Espaco.bloco - 4)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(nsColor: .textBackgroundColor),
                            in: RoundedRectangle(cornerRadius: Raio.pequeno))
                .overlay {
                    RoundedRectangle(cornerRadius: Raio.pequeno)
                        .strokeBorder(.separator)
                }
        }
    }

    private var comandoDeTesteTexto: String {
        "swift run -c release llm-runner --model \(runner.outDir)/bundle --prompt \"Olá\""
    }

    private func copyMarkdown() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(reportMarkdown, forType: .string)
        isCopied = true
        Task {
            try? await Task.sleep(for: .seconds(2))
            isCopied = false
        }
    }

    private func formatDuration(_ duration: Duration) -> String {
        let totalSeconds = Int(duration.components.seconds)
        let minutes = totalSeconds / 60
        let seconds = totalSeconds % 60
        if minutes > 0 {
            return String(format: "%d min %02d s", minutes, seconds)
        }
        return String(format: "%d s", seconds)
    }
}

public struct ReportDocument: FileDocument {
    public static var readableContentTypes: [UTType] { [.plainText] }
    public var text: String

    public init(text: String) {
        self.text = text
    }

    public init(configuration: ReadConfiguration) throws {
        if let data = configuration.file.regularFileContents {
            text = String(decoding: data, as: UTF8.self)
        } else {
            text = ""
        }
    }

    public func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let data = Data(text.utf8)
        return FileWrapper(regularFileWithContents: data)
    }
}
