import Foundation

public struct BinaryResolver: Sendable {
    public enum Resolution: Sendable, Equatable {
        case executable(URL)
        case uvWrapper(uvURL: URL, projectDir: URL)
        case notFound(suggestedCommand: String)
    }

    public static func resolve(projectDirectory: URL? = nil) -> Resolution {
        let fileManager = FileManager.default
        let home = fileManager.homeDirectoryForCurrentUser

        // 1. ~/.local/bin/aguardente (padrão uv tool install)
        let localBin = home.appendingPathComponent(".local/bin/aguardente")
        if fileManager.isExecutableFile(atPath: localBin.path) {
            return .executable(localBin)
        }

        // 2. /opt/homebrew/bin/aguardente
        let homebrewBin = URL(fileURLWithPath: "/opt/homebrew/bin/aguardente")
        if fileManager.isExecutableFile(atPath: homebrewBin.path) {
            return .executable(homebrewBin)
        }

        // 3. /usr/local/bin/aguardente
        let usrLocalBin = URL(fileURLWithPath: "/usr/local/bin/aguardente")
        if fileManager.isExecutableFile(atPath: usrLocalBin.path) {
            return .executable(usrLocalBin)
        }

        // 4. Project .venv/bin/aguardente
        if let projectDir = projectDirectory {
            let venvBin = projectDir.appendingPathComponent(".venv/bin/aguardente")
            if fileManager.isExecutableFile(atPath: venvBin.path) {
                return .executable(venvBin)
            }
        }

        // 5. PATH resolution via environment
        if let pathEnv = ProcessInfo.processInfo.environment["PATH"] {
            for dir in pathEnv.split(separator: ":") {
                let candidate = URL(fileURLWithPath: String(dir)).appendingPathComponent("aguardente")
                if fileManager.isExecutableFile(atPath: candidate.path) {
                    return .executable(candidate)
                }
            }
        }

        // 6. Check for `uv` in common locations
        let uvLocations = [
            home.appendingPathComponent(".local/bin/uv"),
            home.appendingPathComponent(".cargo/bin/uv"),
            URL(fileURLWithPath: "/opt/homebrew/bin/uv"),
            URL(fileURLWithPath: "/usr/local/bin/uv")
        ]
        for uvLoc in uvLocations {
            if fileManager.isExecutableFile(atPath: uvLoc.path) {
                let project = projectDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                return .uvWrapper(uvURL: uvLoc, projectDir: project)
            }
        }

        return .notFound(suggestedCommand: "uv tool install aguardente  # ou execute em modo de desenvolvimento com uv")
    }
}
