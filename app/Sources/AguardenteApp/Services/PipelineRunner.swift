import Foundation
import SwiftUI
import AppKit

@Observable @MainActor
public final class PipelineRunner {
    public enum Phase: Equatable {
        case idle
        case running
        case paused
        case cancelled
        case finished(ok: Bool)
        case failed(String)

        public var description: String {
            switch self {
            case .idle: "Pronto para começar"
            case .running: "Executando"
            case .paused: "Pausado"
            case .cancelled: "Interrompido"
            case .finished(let ok): ok ? "Concluído" : "Concluído com avisos"
            case .failed(let err): "Falhou: \(err)"
            }
        }

        /// Há um processo vivo — suspenso ou não — que pode ser interrompido.
        public var isActive: Bool {
            switch self {
            case .running, .paused: true
            default: false
            }
        }
    }

    public private(set) var stages: [Stage] = []
    public private(set) var activeStageID: Stage.ID?
    public private(set) var previousStageID: Stage.ID?
    public private(set) var phase: Phase = .idle
    public private(set) var elapsed: Duration = .zero

    public var modelName: String {
        didSet { Preferences.modelName = modelName }
    }

    /// Destino absoluto. Um caminho relativo era resolvido contra o diretório
    /// de trabalho, que num `.app` aberto pelo Finder é `/` — o pipeline
    /// tentava gravar na raiz do disco e falhava por permissão.
    public var outDir: String {
        didSet { Preferences.outDir = outDir }
    }

    /// Alvo de parâmetros, ou `nil` para deixar o pipeline dimensionar pela RAM.
    ///
    /// Enviar um número fixo era o defeito mais grave da interface: 1,44 B foi
    /// calibrado para um modelo e uma máquina, e ia junto para qualquer outro
    /// modelo escolhido. Numa máquina cujo teto de treino é menor, todo `run`
    /// abortava antes de baixar coisa alguma. Sem `--target-params` o pipeline
    /// escolhe o alvo que cabe, para o modelo que o usuário pediu de fato.
    public var targetParams: Double? {
        didSet { Preferences.targetParams = targetParams }
    }

    /// Verdadeiro quando a execução está em curso mas nada chega há tempo demais.
    ///
    /// Sem este sinal, um pipeline travado deixava a interface indicando
    /// "Executando" indefinidamente, sem diferença visível de um que progride.
    public private(set) var isStalled: Bool = false

    /// Intervalo sem eventos a partir do qual a execução é dada como sem
    /// resposta. Etapas longas — download, treino — passam minutos sem emitir
    /// nada, então o limiar é generoso de propósito.
    private static let limiteSemResposta: Duration = .seconds(300)

    private var process: Process?
    private var streamTask: Task<Void, Never>?
    private var timerTask: Task<Void, Never>?
    private var startTime: ContinuousClock.Instant?
    private var accumulatedDuration: Duration = .zero
    private var lastEventAt: ContinuousClock.Instant?

    public var isRunning: Bool {
        phase == .running
    }

    public var completedCount: Int {
        stages.filter { $0.state == .ok || $0.state == .skipped }.count
    }

    /// Há saída em disco que valha revelar no Finder.
    public var hasOutput: Bool {
        FileManager.default.fileExists(atPath: (outDir as NSString).expandingTildeInPath)
    }

    public init() {
        modelName = Preferences.modelName
        outDir = Preferences.outDir
        targetParams = Preferences.targetParams
        loadDefaultStages()
    }

    public func stage(_ id: Stage.ID?) -> Stage? {
        guard let id else { return nil }
        return stages.first { $0.id == id }
    }

    public func loadDefaultStages() {
        stages = [
            Stage(id: "fetch", title: "Download", rationale: "Baixa os pesos originais do modelo e tokenizador via aria2c."),
            Stage(id: "prune", title: "Poda estruturada", rationale: "Corta camadas e canais do próprio modelo até atingir o tamanho-alvo dimensionado pela RAM."),
            Stage(id: "logits", title: "Geração de logits", rationale: "Gera saídas de calibração do modelo professor para orientar o processo de destilação."),
            Stage(id: "recover", title: "Recuperação", rationale: "Treino de destilação para recuperar a perplexidade perdida na poda estruturada."),
            Stage(id: "export", title: "Conversão Core AI", rationale: "Converte e quantiza o modelo para execução acelerada via Apple Core AI no Neural Engine / GPU.")
        ]
        activeStageID = nil
        previousStageID = nil
        phase = .idle
        elapsed = .zero
        accumulatedDuration = .zero
    }

    public func start() {
        switch phase {
        case .running:
            return
        case .paused:
            resume()
            return
        case .idle:
            break
        case .cancelled, .finished, .failed:
            reset() // Uma execução encerrada recomeça do zero, com as etapas limpas.
        }

        phase = .running
        startTime = ContinuousClock.now
        lastEventAt = ContinuousClock.now
        isStalled = false
        startTimer()

        let resolution = BinaryResolver.resolve()
        startProcess(with: resolution)
    }

    public func toggle() {
        if isRunning {
            pause()
        } else {
            start()
        }
    }

    public func pause() {
        guard isRunning else { return }
        phase = .paused
        stopTimer()
        process?.suspend()
    }

    public func resume() {
        guard phase == .paused else { return }
        phase = .running
        startTime = ContinuousClock.now
        startTimer()
        process?.resume()
    }

    public func cancel() {
        streamTask?.cancel()
        streamTask = nil
        let estavaPausado = phase == .paused
        stopTimer()
        if let proc = process, proc.isRunning {
            // SIGTERM não chega a um processo parado por SIGSTOP. Sem retomar
            // antes, cancelar durante a pausa deixava o pipeline suspenso para
            // sempre, segurando a RAM e o modelo já carregado.
            if estavaPausado {
                proc.resume()
            }
            proc.terminate()
        }
        process = nil
        phase = .cancelled
    }

    public func reset() {
        cancel()
        loadDefaultStages()
    }

    /// Abre o destino no Finder, e só ele.
    ///
    /// O recuo anterior era o diretório de trabalho do processo — `/` num app
    /// aberto pelo Finder. Revelar a raiz do disco como resposta a um clique
    /// não ajuda ninguém; sem saída, a ação simplesmente não faz nada, e a
    /// interface a mantém desabilitada.
    public func revealOutput() {
        let path = (outDir as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: path) else { return }
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: path)
    }

    private func startTimer() {
        timerTask?.cancel()
        timerTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                guard let self, self.phase == .running, let start = self.startTime else { break }
                let agora = ContinuousClock.now
                self.elapsed = self.accumulatedDuration + (agora - start)
                if let ultimo = self.lastEventAt {
                    self.isStalled = (agora - ultimo) > PipelineRunner.limiteSemResposta
                }
            }
        }
    }

    private func stopTimer() {
        if let start = startTime {
            accumulatedDuration += ContinuousClock.now - start
            startTime = nil
        }
        timerTask?.cancel()
        timerTask = nil
    }

    private func startProcess(with resolution: BinaryResolver.Resolution) {
        streamTask?.cancel()

        let proc = Process()
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env

        // O destino é criado aqui para que exista antes de qualquer download, e
        // o diretório de trabalho do processo passa a ser o pai dele. Sem isso o
        // filho herdava `/` num app aberto pelo Finder.
        let destino = URL(fileURLWithPath: (outDir as NSString).expandingTildeInPath)
        try? FileManager.default.createDirectory(at: destino, withIntermediateDirectories: true)

        switch resolution {
        case .executable(let url):
            proc.executableURL = url
            proc.currentDirectoryURL = destino.deletingLastPathComponent()
            proc.arguments = argumentos(destino: destino.path)

        case .uvWrapper(let uvURL, let projectDir):
            proc.executableURL = uvURL
            // `uv run` resolve o projeto a partir do diretório de trabalho, então
            // aqui ele precisa ser o do repositório, não o do destino.
            proc.currentDirectoryURL = projectDir
            proc.arguments = ["run", "aguardente"] + argumentos(destino: destino.path)

        case .notFound(let hint):
            let errorMsg = "Binário aguardente não encontrado. Execute: \(hint)"
            stage("fetch")?.appendCappedLog(LogLine(text: errorMsg, level: "error"))
            phase = .failed("Binário aguardente não encontrado")
            return
        }

        // Lançar antes de publicar a referência. Com `run()` dentro da tarefa
        // destacada, um `cancel()` no intervalo chamava `terminate()` num
        // processo ainda não lançado — exceção do Objective-C, que derruba o app.
        do {
            try proc.run()
        } catch {
            handleProcessError(error)
            return
        }
        self.process = proc

        streamTask = Task.detached { [weak self] in
            let lines = pipe.fileHandleForReading.bytes.lines
            let decoder = JSONDecoder()
            do {
                for try await line in lines {
                    guard let data = line.data(using: .utf8) else { continue }
                    if let event = try? decoder.decode(PipelineEvent.self, from: data) {
                        await self?.apply(event)
                    } else if !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        await self?.apply(.log(id: "pipeline", message: line, level: "info"))
                    }
                }
            } catch {
                await self?.apply(.log(id: "pipeline",
                                       message: "leitura da saída interrompida: \(error.localizedDescription)",
                                       level: "warn"))
            }

            proc.waitUntilExit()
            let exitCode = proc.terminationStatus
            await self?.handleProcessTermination(exitCode: exitCode)
        }
    }

    /// Argumentos do `aguardente run`.
    ///
    /// `--target-params` só entra quando o usuário pediu um alvo. Omitido, o
    /// pipeline dimensiona pela RAM da máquina e pelo modelo escolhido, que é
    /// o comportamento correto para qualquer modelo — inclusive os que não são
    /// o padrão.
    func argumentos(destino: String) -> [String] {
        var args = ["run", modelName, "-o", destino, "--json"]
        if let alvo = targetParams, alvo > 0 {
            args.append(contentsOf: ["--target-params", String(format: "%.0f", alvo)])
        }
        return args
    }

    /// Conclui a execução a partir do código de saída do processo.
    ///
    /// Só quando não há nada melhor. O código de saída é a informação mais
    /// pobre disponível: uma parada deliberada já registrou `.cancelled`, e um
    /// evento `error` já trouxe a mensagem específica com a dica. Sem esta
    /// guarda, o encerramento do processo sobrescrevia as duas — cancelar
    /// mostrava "Falhou: processo encerrou com código 15", e o diagnóstico que
    /// o pipeline mandou desaparecia atrás do código de saída.
    /// Ponto de entrada do teste para o encerramento do processo.
    ///
    /// Os testes não criam processo real, e é justamente no encerramento que
    /// estava o defeito: sem uma porta de entrada, a guarda ficaria sem
    /// cobertura.
    func concluirParaTeste(exitCode: Int32) {
        handleProcessTermination(exitCode: exitCode)
    }

    private func handleProcessTermination(exitCode: Int32) {
        stopTimer()
        guard phase.isActive else { return }
        phase = exitCode == 0
            ? .finished(ok: true)
            : .failed("Processo encerrou com código \(exitCode)")
    }

    private func handleProcessError(_ error: Error) {
        stopTimer()
        phase = .failed(error.localizedDescription)
    }

    public func apply(_ event: PipelineEvent) {
        lastEventAt = ContinuousClock.now
        isStalled = false

        switch event {
        case .plan(let planStages):
            if !planStages.isEmpty {
                self.stages = planStages.map { p in
                    Stage(id: p.id, title: p.title, rationale: p.rationale ?? "")
                }
            }

        case .stageStart(let id, _, _, let rationale):
            previousStageID = activeStageID
            activeStageID = id
            if let st = stage(id) {
                st.state = .running
                if !rationale.isEmpty { st.rationale = rationale }
            }

        case .stageEnd(let id, let ok, let ms):
            if let st = stage(id) {
                st.state = ok ? .ok : .failed
                st.duration = .milliseconds(ms)
            }

        case .progress(let id, let cur, let tot):
            if let st = stage(id) {
                if let tot, tot > 0 {
                    st.progress = Double(cur) / Double(tot)
                } else {
                    st.progress = nil
                }
            }

        case .metric(let id, let key, let value, let unit):
            if let st = stage(id) {
                let meetsTarget = !(key.contains("ppl") && value > 25.0)
                let metric = Metric(key: key, value: value, unit: unit, meetsTarget: meetsTarget)
                if let idx = st.metrics.firstIndex(where: { $0.key == key }) {
                    st.metrics[idx] = metric
                } else {
                    st.metrics.append(metric)
                }
            }

        case .log(let id, let message, let level):
            let line = LogLine(text: message, level: level)
            if let st = stage(id) {
                st.appendCappedLog(line)
            } else if let active = stage(activeStageID) {
                active.appendCappedLog(line)
            } else if let first = stages.first {
                first.appendCappedLog(line)
            }

        case .error(let id, let message, let hint):
            let formattedMsg = hint != nil ? "\(message) (Dica: \(hint!))" : message
            let line = LogLine(text: formattedMsg, level: "error")
            if let st = stage(id) {
                st.state = .failed
                st.appendCappedLog(line)
            }
            phase = .failed(message)

        case .unknown:
            break
        }
    }

    /// Carrega dados de demonstração com métricas e logs para testes visuais.
    public func loadDemoStages() {
        stages = [
            Stage(
                id: "fetch",
                title: "Download",
                rationale: "Baixa os pesos originais do modelo e tokenizador via aria2c.",
                state: .ok,
                progress: 1.0,
                metrics: [
                    Metric(key: "bytes", label: "Tamanho Baixado", value: 8.42, unit: "B", meetsTarget: true)
                ],
                log: [
                    LogLine(text: "aria2c: conectando a huggingface.co/Qwen/Qwen3-4B", level: "info"),
                    LogLine(text: "8 conexões paralelas estabelecidas", level: "info"),
                    LogLine(text: "download concluído: 8.42 GB em 14s", level: "info")
                ],
                duration: .seconds(14)
            ),
            Stage(
                id: "prune",
                title: "Poda estruturada",
                rationale: "Corta camadas e canais do próprio modelo até atingir o tamanho-alvo. O que sai é escolhido por importância medida, não por posição.",
                state: .ok,
                progress: 1.0,
                metrics: [
                    Metric(key: "params_pruned", label: "Parâmetros Restantes", value: 1.44, unit: "B", meetsTarget: true)
                ],
                log: [
                    LogLine(text: "avaliando importância com 32 lotes de calibração", level: "info"),
                    LogLine(text: "fatiando mlp.gate_proj camada 14", level: "info"),
                    LogLine(text: "fatiando mlp.up_proj camada 14", level: "info"),
                    LogLine(text: "removendo bloco 15 (importância 0.031)", level: "info"),
                    LogLine(text: "poda concluída: 4.12 B → 1.44 B parâmetros", level: "info")
                ],
                duration: .seconds(42)
            ),
            Stage(
                id: "logits",
                title: "Geração de logits",
                rationale: "Gera saídas de calibração do modelo professor para orientar o processo de destilação.",
                state: .ok,
                progress: 1.0,
                metrics: [
                    Metric(key: "ppl_teacher", label: "PPL Original", value: 8.41, meetsTarget: true),
                    Metric(key: "shards", label: "Shards Gerados", value: 64, meetsTarget: true)
                ],
                log: [
                    LogLine(text: "perplexidade do teacher: 8.41", level: "info"),
                    LogLine(text: "gerando logits top-128 para 256 lotes", level: "info"),
                    LogLine(text: "64 shards gravados em disco", level: "info")
                ],
                duration: .seconds(118)
            ),
            Stage(
                id: "recover",
                title: "Recuperação",
                rationale: "Treino de destilação para recuperar a perplexidade perdida na poda estruturada.",
                state: .running,
                progress: 0.65,
                metrics: [
                    Metric(key: "ppl_pruned", label: "PPL Pós-Poda", value: 19.82, meetsTarget: false),
                    Metric(key: "ppl_recovered", label: "PPL Atual", value: 9.15, meetsTarget: true),
                    Metric(key: "recovered_fraction", label: "Queda Recuperada", value: 0.935, meetsTarget: true)
                ],
                log: [
                    LogLine(text: "perplexidade pós-poda: 19.82", level: "warn"),
                    LogLine(text: "iniciando destilação: 2 épocas, lr 3e-05, T 2.0", level: "info"),
                    LogLine(text: "passo 50/100 · loss 0.4128", level: "info"),
                    LogLine(text: "passo 65/100 · loss 0.3842", level: "info")
                ],
                duration: .seconds(252)
            ),
            Stage(
                id: "export",
                title: "Conversão Core AI",
                rationale: "Converte e quantiza o modelo para execução acelerada via Apple Core AI no Neural Engine / GPU.",
                state: .pending,
                progress: nil,
                metrics: [],
                log: [],
                duration: nil
            )
        ]
        activeStageID = "recover"
        previousStageID = "logits"
        phase = .running
        elapsed = .seconds(426)
    }
}
