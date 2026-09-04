import SwiftUI
import AppKit

/// Preferências da execução: o que converter, para onde, e com que alvo.
///
/// Estes três valores eram constantes no código, o que prendia o aplicativo a
/// um único modelo. O alvo tem um modo automático porque é o correto na maior
/// parte dos casos: o pipeline mede a RAM da máquina e escolhe o maior tamanho
/// que ainda treina nela. Um número fixo, ao contrário, vale para uma máquina
/// só — e aborta a execução em todas as outras.
public struct SettingsView: View {
    @Bindable var runner: PipelineRunner
    @State private var escolhendoDestino = false
    @State private var alvoAutomatico: Bool
    @State private var alvoEmBilhoes: Double

    public init(runner: PipelineRunner) {
        self.runner = runner
        let alvo = runner.targetParams
        _alvoAutomatico = State(initialValue: alvo == nil)
        _alvoEmBilhoes = State(initialValue: (alvo ?? 1.44e9) / 1e9)
    }

    public var body: some View {
        Form {
            // A explicação de cada grupo vira rodapé da própria seção em vez de
            // um `Text` solto entre os controles: num `Form` agrupado o rodapé
            // já recebe o tamanho, a cor e o recuo certos, e o texto deixa de
            // ser lido como mais um campo do formulário.
            Section {
                TextField("Modelo", text: $runner.modelName, prompt: Text("organização/nome"))
            } header: {
                Text("Modelo de origem")
            } footer: {
                // Sem o `frame`, o rodapé de uma `Section` num `Form` agrupado
                // de macOS sai justificado à direita e lê como legenda solta.
                rodape("Identificador do Hugging Face, URL do Hugging Face ou do GitHub, "
                       + "ou o caminho de um diretório local.")
            }

            Section {
                LabeledContent("Destino") {
                    HStack(spacing: Espaco.interno) {
                        Text(runner.outDir)
                            .lineLimit(1)
                            .truncationMode(.head)
                            .foregroundStyle(.secondary)
                            .help(runner.outDir)
                        Spacer(minLength: 0)
                        Button("Escolher…") { escolhendoDestino = true }
                    }
                }
            } header: {
                Text("Destino")
            } footer: {
                rodape("Os pesos baixados, os logits e o pacote final são gravados aqui. "
                       + "Reserve dezenas de gigabytes.")
            }

            Section {
                Toggle("Dimensionar pela RAM desta máquina", isOn: $alvoAutomatico)
                    .onChange(of: alvoAutomatico) { _, automatico in
                        runner.targetParams = automatico ? nil : alvoEmBilhoes * 1e9
                    }

                if !alvoAutomatico {
                    LabeledContent("Alvo") {
                        HStack(spacing: Espaco.interno) {
                            Slider(value: $alvoEmBilhoes, in: 0.1...14, step: 0.01)
                                .onChange(of: alvoEmBilhoes) { _, novo in
                                    runner.targetParams = novo * 1e9
                                }
                            Text(String(format: "%.2f B", alvoEmBilhoes))
                                .font(.body.monospacedDigit())
                                .frame(width: 68, alignment: .trailing)
                        }
                    }
                }
            } header: {
                Text("Tamanho do resultado")
            } footer: {
                rodape(alvoAutomatico
                       ? "O pipeline escolhe o maior modelo que ainda treina na memória disponível."
                       : "Um alvo acima do teto de treino da máquina interrompe a execução antes do download.")
            }
        }
        .formStyle(.grouped)
        .frame(width: 480)
        .fixedSize(horizontal: false, vertical: true)
        .fileImporter(isPresented: $escolhendoDestino,
                      allowedContentTypes: [.folder]) { resultado in
            if case .success(let url) = resultado {
                runner.outDir = url.path
            }
        }
    }

    private func rodape(_ texto: String) -> some View {
        Text(texto)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }
}
