// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AguardenteApp",
    // A interface e a destilação rodam a partir do macOS 14. A conversão Core AI
    // exige uma versão bem mais nova, e por isso é a única parte com barreira
    // própria em tempo de execução — ver `CoreAIDisponibilidade`.
    platforms: [
        .macOS("14.0")
    ],
    products: [
        .executable(
            name: "aguardente-app",
            targets: ["AguardenteApp"]
        )
    ],
    targets: [
        .executableTarget(
            name: "AguardenteApp",
            path: "app/Sources/AguardenteApp",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "AguardenteAppTests",
            dependencies: ["AguardenteApp"],
            path: "app/Tests/AguardenteAppTests"
        )
    ]
)
