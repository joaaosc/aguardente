import SwiftUI

public struct StageDetail: View {
    let stage: Stage
    let isLogVisible: Bool

    public init(stage: Stage, isLogVisible: Bool) {
        self.stage = stage
        self.isLogVisible = isLogVisible
    }

    public var body: some View {
        VSplitView {
            overview

            if isLogVisible {
                LogPane(lines: stage.log)
                    .frame(minHeight: 140, idealHeight: 230)
            }
        }
        .background(.background)
    }

    private var overview: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(stage.title)
                        .font(.title2.weight(.semibold))

                    if !stage.rationale.isEmpty {
                        Text(stage.rationale)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }

                if stage.state == .running, let progress = stage.progress {
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                }

                if stage.state == .failed {
                    Label("Esta etapa falhou. O motivo está no log.", systemImage: "exclamationmark.triangle.fill")
                        .font(.callout)
                        .foregroundStyle(.orange)
                }

                if !stage.metrics.isEmpty {
                    StageMetrics(metrics: stage.metrics)
                }
            }
            .padding(24)
            .frame(maxWidth: 620, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minHeight: 200)
    }
}
