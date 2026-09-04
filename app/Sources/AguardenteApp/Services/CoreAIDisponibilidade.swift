import Foundation

/// Disponibilidade da conversão Core AI no sistema em execução.
///
/// A destilação — download, poda, geração de logits e recuperação — roda a
/// partir do macOS 14: é PyTorch puro, e não depende de nada da Apple além do
/// Metal. A etapa final é diferente: converter e compilar para `.aimodel`
/// depende da stack Core AI, que exige um sistema bem mais novo.
///
/// A distinção existe para que o aplicativo abra e seja útil num Mac com
/// macOS 14 — destilando um modelo e parando antes do empacotamento — em vez
/// de recusar a instalação inteira por causa da última etapa.
public enum CoreAIDisponibilidade {
    /// Primeira versão do macOS em que a conversão Core AI é suportada.
    public static let versaoMinima = OperatingSystemVersion(
        majorVersion: 27, minorVersion: 0, patchVersion: 0)

    public static var suportada: Bool {
        ProcessInfo.processInfo.isOperatingSystemAtLeast(versaoMinima)
    }

    /// Explicação para quem está num sistema anterior.
    public static var motivo: String {
        let atual = ProcessInfo.processInfo.operatingSystemVersion
        return "A conversão Core AI exige macOS \(versaoMinima.majorVersion) ou mais recente; "
            + "este Mac roda o macOS \(atual.majorVersion).\(atual.minorVersion). "
            + "As demais etapas funcionam normalmente e produzem o modelo destilado."
    }

    /// Identificador da etapa do pipeline que depende do Core AI.
    public static let etapaDependente = "export"
}
