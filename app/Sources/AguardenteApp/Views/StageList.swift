import SwiftUI

public struct StageList: View {
    let runner: PipelineRunner
    @Binding var selection: Stage.ID?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    public init(runner: PipelineRunner, selection: Binding<Stage.ID?>) {
        self.runner = runner
        self._selection = selection
    }

    public var body: some View {
        ScrollViewReader { proxy in
            List(runner.stages, selection: $selection) { stage in
                HStack {
                    Image(systemName: stage.state.symbol)
                        .foregroundStyle(stage.state.tint)
                        .symbolEffect(.pulse, isActive: stage.state == .running && !reduceMotion)

                    Text(stage.title)
                        .font(.body)
                        .foregroundStyle(stage.state == .pending ? .secondary : .primary)

                    Spacer()

                    if let duration = stage.duration {
                        Text(formatDuration(duration))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.vertical, 3)
                .id(stage.id)
                .accessibilityElement(children: .combine)
                .accessibilityLabel("\(stage.title), \(stage.state.labelDescription)\(stage.duration.map { ", \(formatDuration($0))" } ?? "")")
            }
            .listStyle(.sidebar)
            .onChange(of: runner.activeStageID) { _, new in
                guard let new, selection == nil || selection == runner.previousStageID else {
                    return // Não sequestra a navegação se o usuário escolheu outra manualmente
                }
                withAnimation(.easeInOut(duration: 0.3)) {
                    proxy.scrollTo(new, anchor: .center)
                }
            }
        }
    }

    private func formatDuration(_ duration: Duration) -> String {
        let totalSeconds = Int(duration.components.seconds)
        let minutes = totalSeconds / 60
        let seconds = totalSeconds % 60
        if minutes > 0 {
            return String(format: "%02d:%02d", minutes, seconds)
        }
        return String(format: "%ds", seconds)
    }
}
