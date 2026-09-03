import SwiftUI

/// Números coletados por uma etapa, como pares rótulo-valor agrupados.
///
/// Uma caixa só para todas as métricas, em vez de um cartão por número: o valor
/// que interessa é a comparação entre eles, e alinhá-los à direita torna essa
/// leitura imediata.
public struct StageMetrics: View {
    public let metrics: [Metric]

    public init(metrics: [Metric]) {
        self.metrics = metrics
    }

    public var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(metrics) { metric in
                    LabeledContent {
                        HStack(spacing: 4) {
                            Text(verbatim: metric.formatted)
                                .font(.body.monospacedDigit())
                                .foregroundStyle(metric.meetsTarget ? Color.primary : Color.orange)

                            if !metric.meetsTarget {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .font(.caption)
                                    .foregroundStyle(Color.orange)
                            }
                        }
                    } label: {
                        Text(verbatim: metric.label)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel(Text(verbatim: metric.label))
                    .accessibilityValue(Text(verbatim: metric.meetsTarget ? metric.formatted : "\(metric.formatted), fora da meta"))
                }
            }
            .padding(.vertical, 2)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Text("Métricas")
        }
    }
}
