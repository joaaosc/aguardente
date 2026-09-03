import Testing
import Foundation
@testable import AguardenteApp

@Suite("PipelineRunner Tests")
@MainActor
struct PipelineRunnerTests {

    @Test("Aplica eventos sequencialmente no PipelineRunner")
    func testApplyEvents() {
        let runner = PipelineRunner()
        #expect(runner.stages.count == 5)
        #expect(runner.phase == .idle)

        // 1. Início de etapa
        runner.apply(.stageStart(id: "fetch", n: 1, of: 5, rationale: "Baixando"))
        #expect(runner.activeStageID == "fetch")
        #expect(runner.stage("fetch")?.state == .running)
        #expect(runner.stage("fetch")?.rationale == "Baixando")

        // 2. Progresso
        runner.apply(.progress(id: "fetch", current: 50, total: 100))
        #expect(runner.stage("fetch")?.progress == 0.5)

        // 3. Log
        runner.apply(.log(id: "fetch", message: "Conectado", level: "info"))
        #expect(runner.stage("fetch")?.log.count == 1)
        #expect(runner.stage("fetch")?.log.first?.text == "Conectado")

        // 4. Métrica
        runner.apply(.metric(id: "fetch", key: "bytes", value: 8.5e9, unit: "B"))
        #expect(runner.stage("fetch")?.metrics.count == 1)

        // 5. Conclusão da etapa
        runner.apply(.stageEnd(id: "fetch", ok: true, ms: 1500))
        #expect(runner.stage("fetch")?.state == .ok)
        #expect(runner.stage("fetch")?.duration == .milliseconds(1500))
        #expect(runner.completedCount == 1)
    }

    @Test("Alternar a partir de pausado retoma a execução em vez de reiniciá-la")
    func testTogglePausedResumes() {
        let runner = PipelineRunner()
        runner.loadDemoStages()
        #expect(runner.isRunning)

        runner.toggle()
        #expect(runner.phase == .paused)

        runner.toggle()
        #expect(runner.phase == .running)

        // As etapas já concluídas sobrevivem: retomar não é recomeçar do zero.
        #expect(runner.stages.count == 5)
        #expect(runner.stage("prune")?.state == .ok)
        #expect(runner.activeStageID == "recover")
    }

    @Test("Parar registra interrupção deliberada, não falha")
    func testCancelIsNotAFailure() {
        let runner = PipelineRunner()
        runner.loadDemoStages()

        runner.cancel()
        #expect(runner.phase == .cancelled)
        #expect(runner.phase.description == "Interrompido")
        #expect(runner.phase.isActive == false)
    }

    @Test("isActive distingue processo vivo de execução encerrada")
    func testPhaseIsActive() {
        #expect(PipelineRunner.Phase.running.isActive)
        #expect(PipelineRunner.Phase.paused.isActive)
        #expect(!PipelineRunner.Phase.idle.isActive)
        #expect(!PipelineRunner.Phase.cancelled.isActive)
        #expect(!PipelineRunner.Phase.finished(ok: true).isActive)
        #expect(!PipelineRunner.Phase.failed("erro").isActive)
    }

    @Test("Carregamento de dados de demonstração")
    func testDemoData() {
        let runner = PipelineRunner()
        runner.loadDemoStages()

        #expect(runner.stages.count == 5)
        #expect(runner.activeStageID == "recover")
        #expect(runner.isRunning == true)
        #expect(runner.stage("prune")?.state == .ok)
        #expect(runner.stage("recover")?.state == .running)
        #expect(runner.stage("export")?.state == .pending)
    }
}
