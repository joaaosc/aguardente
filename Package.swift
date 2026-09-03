// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AguardenteApp",
    platforms: [
        .macOS("26.0")
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
