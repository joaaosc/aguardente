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
        md += "```bash\nswift run -c release llm-runner --model \(runner.outDir)/bundle --prompt \"Olá\"\n```\n"
        return md
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Cabeçalho
                    VStack(alignment: .leading, spacing: 6) {
                        Text(runner.modelName)
                            .font(.title2.weight(.bold))
                        Text("Pipeline de poda estruturada, destilação e conversão Core AI")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.bottom, 8)

                    Divider()

                    // Resumo das Etapas
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Etapas")
                            .font(.headline)

                        ForEach(runner.stages) { stage in
                            HStack {
                                Image(systemName: stage.state.symbol)
                                    .foregroundStyle(stage.state.tint)
                                Text(stage.title)
                                    .fontWeight(.medium)
                                Spacer()
                                Text(stage.duration.map { formatDuration($0) } ?? "—")
                                    .font(.callout.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }

                    Divider()

                    // Métricas, agrupadas por etapa. A identidade de `Metric` é
                    // a chave, única dentro de uma etapa mas não entre elas:
                    // achatar tudo numa lista só daria ids repetidos ao ForEach,
                    // que descartaria linhas em silêncio.
                    if runner.stages.contains(where: { !$0.metrics.isEmpty }) {
                        VStack(alignment: .leading, spacing: 14) {
                            Text("Métricas Coletadas")
                                .font(.headline)

                            ForEach(runner.stages.filter { !$0.metrics.isEmpty }) { etapa in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(etapa.title)
                                        .font(.subheadline.weight(.medium))
                                        .foregroundStyle(.secondary)

                                    ForEach(etapa.metrics) { m in
                                        HStack {
                                            Text(m.label)
                                            Spacer()
                                            Text(m.formatted)
                                                .font(.callout.monospacedDigit().weight(.semibold))
                                                .foregroundStyle(m.meetsTarget ? Color.primary : Color.orange)
                                        }
                                        .padding(.vertical, 2)
                                    }
                                }
                            }
                        }

                        Divider()
                    }

                    // Comando de Teste
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Comando para testar no Terminal")
                            .font(.headline)

                        Text("swift run -c release llm-runner --model \(runner.outDir)/bundle --prompt \"Olá\"")
                            .font(.system(.caption, design: .monospaced))
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(nsColor: .textBackgroundColor))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .textSelection(.enabled)
                    }
                }
                .padding(24)
            }
            .navigationTitle("Relatório da Execução")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Fechar") { dismiss() }
                }

                ToolbarItemGroup(placement: .primaryAction) {
                    Button {
                        copyMarkdown()
                    } label: {
                        Label(isCopied ? "Copiado!" : "Copiar Markdown", systemImage: isCopied ? "checkmark" : "doc.on.doc")
                    }

                    Button {
                        reportDocument = ReportDocument(text: reportMarkdown)
                        showFileExporter = true
                    } label: {
                        Label("Exportar...", systemImage: "square.and.arrow.up")
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
        .frame(minWidth: 520, minHeight: 480)
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
