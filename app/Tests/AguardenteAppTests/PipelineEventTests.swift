import Testing
import Foundation
@testable import AguardenteApp

@Suite("PipelineEvent Tests")
struct PipelineEventTests {

    @Test("Decodifica evento plan")
    func testDecodePlan() throws {
        let json = """
        {"ts": 1725170000.0, "ev": "plan", "stages": [{"id": "fetch", "title": "Download", "rationale": "Baixa pesos"}]}
        """
        let data = Data(json.utf8)
        let event = try JSONDecoder().decode(PipelineEvent.self, from: data)

        if case .plan(let stages) = event {
            #expect(stages.count == 1)
            #expect(stages[0].id == "fetch")
            #expect(stages[0].title == "Download")
            #expect(stages[0].rationale == "Baixa pesos")
        } else {
            Issue.record("Esperava evento plan")
        }
    }

    @Test("Decodifica evento stage.start com rationale")
    func testDecodeStageStart() throws {
        let json = """
        {"ts": 1725170001.0, "ev": "stage.start", "id": "prune", "n": 2, "of": 5, "rationale": "Poda estruturada"}
        """
        let data = Data(json.utf8)
        let event = try JSONDecoder().decode(PipelineEvent.self, from: data)

        if case .stageStart(let id, let n, let of, let rationale) = event {
            #expect(id == "prune")
            #expect(n == 2)
            #expect(of == 5)
            #expect(rationale == "Poda estruturada")
        } else {
            Issue.record("Esperava evento stage.start")
        }
    }

    @Test("Decodifica evento metric")
    func testDecodeMetric() throws {
        let json = """
        {"ts": 1725170002.0, "ev": "metric", "id": "recover", "k": "ppl_recovered", "v": 8.45, "unit": ""}
        """
        let data = Data(json.utf8)
        let event = try JSONDecoder().decode(PipelineEvent.self, from: data)

        if case .metric(let id, let key, let value, _) = event {
            #expect(id == "recover")
            #expect(key == "ppl_recovered")
            #expect(value == 8.45)
        } else {
            Issue.record("Esperava evento metric")
        }
    }

    @Test("Formatação de métricas com unidades e sem unidades")
    func testMetricFormatting() {
        let metricWithUnit = Metric(key: "bytes", label: "Tamanho", value: 4_200_000_000, unit: "B")
        #expect(metricWithUnit.formatted == "4.20 GB")

        let metricPpl = Metric(key: "ppl_recovered", value: 9.152)
        #expect(metricPpl.formatted == "9.15")

        let metricFraction = Metric(key: "recovered_fraction", value: 0.942)
        #expect(metricFraction.formatted == "94.2%")
    }
}
