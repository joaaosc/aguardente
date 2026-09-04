import SwiftUI

/// Constantes visuais do aplicativo, num lugar só.
///
/// Antes destes tipos cada view escolhia o próprio espaçamento — 10 aqui, 12
/// ali, 20 e 24 na mesma tela. Nenhum valor estava errado isoladamente, mas o
/// conjunto não formava um ritmo, e é essa ausência de ritmo que o olho lê
/// como interface improvisada. Uma escala fechada de quatro valores resolve
/// isso sem exigir disciplina de quem escreve a próxima view.
public enum Espaco {
    /// Entre um rótulo e o valor que ele descreve.
    public static let minimo: CGFloat = 4
    /// Entre itens de uma mesma linha ou de um mesmo bloco.
    public static let interno: CGFloat = 8
    /// Entre blocos relacionados dentro de uma seção.
    public static let bloco: CGFloat = 16
    /// Entre seções, e das seções para a borda da janela.
    public static let secao: CGFloat = 24
}

public enum Raio {
    /// Chips, pills e badges pequenos.
    public static let pequeno: CGFloat = 8
    /// Cartões de conteúdo.
    public static let medio: CGFloat = 12
    /// Superfícies grandes e painéis.
    public static let grande: CGFloat = 16
}

/// Identidade visual do aplicativo.
///
/// A HIG reserva a cor de destaque do sistema para seleção, botões e ícones da
/// barra lateral — o usuário escolhe essa cor, e o aplicativo não deve
/// sequestrá-la. O cobre abaixo é usado só nas superfícies de marca (o
/// cabeçalho da execução), nunca em controle e nunca em estado.
public enum Marca {
    public static let cobre = Color(red: 0.72, green: 0.45, blue: 0.20)
    public static let ambar = Color(red: 0.91, green: 0.64, blue: 0.24)

    /// Gradiente do cabeçalho: quente no canto superior, dissolvendo para o
    /// fundo da janela.
    ///
    /// A primeira versão usava opacidades entre 0,10 e 0,28 e, renderizada,
    /// simplesmente não existia — sobre o fundo escuro da janela o olho não
    /// distinguia o cabeçalho do corpo. Os valores abaixo foram escolhidos
    /// olhando para a captura, não para o código: fortes o bastante para
    /// delimitar a região do título, fracos o bastante para o texto continuar
    /// sendo o elemento mais claro da tela.
    public static var cabecalho: LinearGradient {
        LinearGradient(
            stops: [
                .init(color: cobre.opacity(0.44), location: 0),
                .init(color: ambar.opacity(0.17), location: 0.45),
                .init(color: .clear, location: 1)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
}

extension View {
    /// Liquid Glass quando o sistema o oferece; material padrão quando não.
    ///
    /// O vidro só existe a partir do macOS 26 e o alvo mínimo do aplicativo é o
    /// 14. Em vez de espalhar `if #available` por cada chamada, o gate mora
    /// aqui — e o ramo antigo usa `.regularMaterial`, que produz a mesma
    /// separação entre camada de controle e conteúdo com os meios da época.
    @ViewBuilder
    public func vidro(em forma: some InsettableShape) -> some View {
        if #available(macOS 26.0, *) {
            glassEffect(.regular, in: forma)
        } else {
            background(.regularMaterial, in: forma)
        }
    }

    /// Espelha e desfoca este fundo por baixo da barra lateral.
    ///
    /// É o efeito que separa a camada de controle da de conteúdo sem deixar uma
    /// borda dura entre elas. Aplicar no fundo, e sobrepor o texto *depois*:
    /// só o fundo deve se estender.
    @ViewBuilder
    public func extensaoDeFundo() -> some View {
        if #available(macOS 26.0, *) {
            backgroundExtensionEffect()
        } else {
            self
        }
    }

    /// Transição da barra de ferramentas para o conteúdo que rola sob ela.
    @ViewBuilder
    public func bordaDeRolagem() -> some View {
        if #available(macOS 26.0, *) {
            scrollEdgeEffectStyle(.hard, for: .top)
        } else {
            self
        }
    }
}

/// Arco de progresso com espessura e traço controlados.
///
/// O `ProgressView` circular do sistema não permite escolher a espessura nem o
/// gradiente do traço, e num anel de 22 pt dentro da barra de ferramentas a
/// diferença entre 2 e 4 pontos decide se o número no centro cabe.
public struct AnelDeProgresso: View {
    private let valor: Double
    private let espessura: CGFloat
    private let cor: Color

    public init(valor: Double, espessura: CGFloat = 3, cor: Color = .accentColor) {
        self.valor = min(max(valor, 0), 1)
        self.espessura = espessura
        self.cor = cor
    }

    public var body: some View {
        ZStack {
            Circle()
                .stroke(.quaternary, lineWidth: espessura)

            Circle()
                .trim(from: 0, to: valor)
                .stroke(cor, style: StrokeStyle(lineWidth: espessura, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeOut(duration: 0.3), value: valor)
        }
    }
}
