import SwiftUI

/// Números coletados por uma etapa, um cartão por métrica.
///
/// A versão anterior era uma `GroupBox` com uma linha `rótulo — valor` por
/// métrica, no mesmo corpo de texto do resto da tela. Formatar dados assim é
/// legítimo num formulário, onde o rótulo é a pergunta e o valor é a resposta;
/// aqui o valor *é* o conteúdo — é o resultado de horas de execução — e ficava
/// tipograficamente indistinguível da explicação da etapa logo acima.
///
/// Em cartões, o número recebe o peso que merece e o rótulo vira legenda. E
/// como os cartões têm largura igual numa grade, a comparação entre eles —
/// que é o que se quer olhando para perplexidade antes e depois — passa a ser
/// imediata.
public struct StageMetrics: View {
    public let metrics: [Metric]
    /// Título da grade, ou `nil` quando o contexto já o fornece.
    ///
    /// No relatório cada grupo já é encabeçado pelo nome da etapa, e repetir
    /// "Métricas" logo abaixo produzia dois títulos empilhados sem conteúdo
    /// entre eles.
    public let titulo: String?

    public init(metrics: [Metric], titulo: String? = "Métricas") {
        self.metrics = metrics
        self.titulo = titulo
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: Espaco.interno) {
            if let titulo {
                Text(titulo)
                    .font(.headline)
            }

            LazyVGrid(
                columns: [GridItem(.adaptive(minimum: 150), spacing: Espaco.interno)],
                alignment: .leading,
                spacing: Espaco.interno
            ) {
                ForEach(metrics) { metric in
                    cartao(for: metric)
                }
            }
        }
    }

    private func cartao(for metric: Metric) -> some View {
        VStack(alignment: .leading, spacing: Espaco.minimo) {
            Text(verbatim: metric.label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2, reservesSpace: true)

            HStack(alignment: .firstTextBaseline, spacing: Espaco.minimo) {
                Text(verbatim: metric.formatted)
                    .font(.title3.weight(.semibold).monospacedDigit())
                    .foregroundStyle(metric.meetsTarget ? Color.primary : Color.orange)
                    .contentTransition(.numericText())
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)

                // Fora da meta é informação de estado, e cor sozinha não é um
                // canal — quem não distingue o laranja precisa do símbolo.
                if !metric.meetsTarget {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(Color.orange)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, Espaco.bloco - 4)
        .padding(.vertical, Espaco.interno + 2)
        // Camada de conteúdo: material padrão, não vidro. Vidro em cartão de
        // conteúdo é o erro que a HIG nomeia explicitamente.
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: Raio.medio))
        .overlay {
            RoundedRectangle(cornerRadius: Raio.medio)
                .strokeBorder(metric.meetsTarget ? Color.clear : Color.orange.opacity(0.45))
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text(verbatim: metric.label))
        .accessibilityValue(Text(verbatim: metric.meetsTarget
                                 ? metric.formatted
                                 : "\(metric.formatted), fora da meta"))
    }
}

#Preview("Métricas") {
    StageMetrics(metrics: [
        Metric(key: "ppl_teacher", value: 8.42),
        Metric(key: "ppl_pruned", value: 31.7, meetsTarget: false),
        Metric(key: "recovered_fraction", value: 0.87),
        Metric(key: "bundle_bytes", value: 1.24e9, unit: "B")
    ])
    .padding(Espaco.secao)
    .frame(width: 560)
}
