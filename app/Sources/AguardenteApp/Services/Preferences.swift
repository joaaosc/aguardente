import Foundation

/// Configuração persistida entre execuções.
///
/// Um `.app` não tem linha de comando: o que no CLI são opções, aqui precisa
/// morar em algum lugar e sobreviver ao fechamento da janela. As chaves ficam
/// centralizadas para que a cena `Settings` e o `PipelineRunner` leiam e
/// escrevam exatamente os mesmos valores.
public enum Preferences {
    private static let chaveModelo = "modelo"
    private static let chaveDestino = "destino"
    private static let chaveAlvo = "alvoDeParametros"

    /// Modelo de origem, no formato que o CLI aceita: identificador do Hugging
    /// Face, URL ou diretório local.
    public static var modelName: String {
        get { defaults.string(forKey: chaveModelo) ?? "Qwen/Qwen3-4B" }
        set { defaults.set(newValue, forKey: chaveModelo) }
    }

    /// Destino das execuções, sempre absoluto.
    public static var outDir: String {
        get { defaults.string(forKey: chaveDestino) ?? destinoPadrao }
        set { defaults.set(newValue, forKey: chaveDestino) }
    }

    /// Alvo de parâmetros, ou `nil` para deixar o pipeline dimensionar pela RAM.
    ///
    /// Guardado como `Double` porque `UserDefaults` não distingue "ausente" de
    /// zero em tipos numéricos: zero é a forma de dizer "automático".
    public static var targetParams: Double? {
        get {
            let bruto = defaults.double(forKey: chaveAlvo)
            return bruto > 0 ? bruto : nil
        }
        set { defaults.set(newValue ?? 0, forKey: chaveAlvo) }
    }

    /// Uma pasta do usuário, e não o diretório de trabalho do processo.
    ///
    /// O diretório de trabalho de um app aberto pelo Finder é `/`, onde nada
    /// pode ser gravado. `~/Documents/Aguardente` é previsível, aparece no
    /// Finder e não exige que o usuário escolha nada para a primeira execução.
    public static var destinoPadrao: String {
        let documentos = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        let base = documentos ?? URL(fileURLWithPath: NSHomeDirectory())
        return base.appendingPathComponent("Aguardente").path
    }

    private static var defaults: UserDefaults { .standard }
}
