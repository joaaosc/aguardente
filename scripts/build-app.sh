#!/bin/bash
#
# Monta, assina e (opcionalmente) notariza o Aguardente.app.
#
# O SwiftPM produz um executável, não um bundle: sem Info.plist não há
# identificador, ícone, versão nem assinatura, e sem isso não há distribuição.
# Este script é o passo que falta entre `swift build` e um .app que abre no
# Finder de outra pessoa.
#
# Uso:
#   scripts/build-app.sh                 # monta e assina com Developer ID
#   scripts/build-app.sh --no-sign       # monta sem assinar (teste local)
#   scripts/build-app.sh --notarize      # monta, assina, notariza e grampeia
#
# Notarização exige credenciais guardadas uma vez no chaveiro:
#   xcrun notarytool store-credentials aguardente-notary \
#     --apple-id <seu-apple-id> --team-id 33FPG9442W --password <senha-de-app>

set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"

NOME="Aguardente"
EXECUTAVEL="aguardente-app"
PERFIL_NOTARIZACAO="${AGUARDENTE_NOTARY_PROFILE:-aguardente-notary}"
DESTINO="$RAIZ/dist"
BUNDLE="$DESTINO/$NOME.app"

assinar=1
notarizar=0
for arg in "$@"; do
	case "$arg" in
		--no-sign) assinar=0 ;;
		--notarize) notarizar=1 ;;
		*) echo "opção desconhecida: $arg" >&2; exit 2 ;;
	esac
done

passo() { printf '\n\033[1m==>\033[0m %s\n' "$1"; }

# Resolve a identidade de assinatura para o SHA-1 do certificado, e não para o
# nome. Um chaveiro com o certificado renovado ao lado do antigo tem duas
# entradas de nome idêntico, e o `codesign` recusa a ambiguidade em vez de
# escolher — falha que só aparece na hora de assinar.
resolver_identidade() {
	if [ -n "${AGUARDENTE_SIGN_IDENTITY:-}" ]; then
		printf '%s' "$AGUARDENTE_SIGN_IDENTITY"
		return
	fi
	local achados
	achados="$(security find-identity -v -p codesigning \
		| grep 'Developer ID Application' \
		| sed -E 's/^ *[0-9]+\) ([0-9A-F]{40}).*/\1/')"
	local total
	total="$(printf '%s\n' "$achados" | grep -c . || true)"
	if [ "$total" -eq 0 ]; then
		echo "nenhuma identidade 'Developer ID Application' no chaveiro." >&2
		echo "Crie uma em developer.apple.com e instale-a, ou use --no-sign." >&2
		exit 1
	fi
	if [ "$total" -gt 1 ]; then
		echo "há $total certificados 'Developer ID Application' no chaveiro:" >&2
		security find-identity -v -p codesigning | grep 'Developer ID Application' >&2
		echo >&2
		echo "Escolha um pelo SHA-1 e repita:" >&2
		echo "  AGUARDENTE_SIGN_IDENTITY=<sha1> scripts/build-app.sh" >&2
		exit 1
	fi
	printf '%s' "$achados"
}

# A versão vem do pacote Python, e não de um número repetido no Info.plist. Três
# declarações independentes divergem na primeira correção; uma só não tem como.
versao_do_pacote() {
	sed -n 's/^__version__ = "\(.*\)"/\1/p' src/aguardente/__init__.py | head -1
}

# ---------------------------------------------------------------- compilação

passo "Compilando em release"
swift build -c release --product "$EXECUTAVEL"
BINARIO="$(swift build -c release --product "$EXECUTAVEL" --show-bin-path)/$EXECUTAVEL"
[ -x "$BINARIO" ] || { echo "binário não encontrado em $BINARIO" >&2; exit 1; }

# -------------------------------------------------------------------- ícone

passo "Gerando o ícone a partir de docs/media/icon.svg"
command -v rsvg-convert >/dev/null || {
	echo "rsvg-convert não encontrado. Instale com: brew install librsvg" >&2
	exit 1
}

ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
# O .icns precisa de cada tamanho em 1x e 2x. Rasterizar direto do SVG em cada
# resolução evita o borrão de reescalar um PNG único.
for tamanho in 16 32 128 256 512; do
	rsvg-convert -w "$tamanho" -h "$tamanho" docs/media/icon.svg \
		-o "$ICONSET/icon_${tamanho}x${tamanho}.png"
	rsvg-convert -w "$((tamanho * 2))" -h "$((tamanho * 2))" docs/media/icon.svg \
		-o "$ICONSET/icon_${tamanho}x${tamanho}@2x.png"
done

# --------------------------------------------------------------- montagem

passo "Montando $BUNDLE"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"

cp "$BINARIO" "$BUNDLE/Contents/MacOS/$EXECUTAVEL"
cp app/Resources/Info.plist "$BUNDLE/Contents/Info.plist"
iconutil -c icns "$ICONSET" -o "$BUNDLE/Contents/Resources/AppIcon.icns"
printf 'APPL????' > "$BUNDLE/Contents/PkgInfo"

VERSAO="$(versao_do_pacote)"
[ -n "$VERSAO" ] || { echo "não foi possível ler __version__ do pacote" >&2; exit 1; }
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSAO" \
	"$BUNDLE/Contents/Info.plist"
# CFBundleVersion precisa crescer a cada envio para a notarização, mesmo quando a
# versão exibida não muda. A contagem de commits é monotônica e não exige
# lembrar de incrementar nada à mão.
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $(git rev-list --count HEAD)" \
	"$BUNDLE/Contents/Info.plist"
echo "versão $VERSAO (build $(git rev-list --count HEAD))"

# As dependências do pipeline são permissivas (BSD-3 e Apache-2.0) e permitem
# redistribuição sob licença restrita, mas exigem que a atribuição viaje junto.
if [ -f THIRD-PARTY-NOTICES.md ]; then
	cp THIRD-PARTY-NOTICES.md "$BUNDLE/Contents/Resources/"
fi
[ -f LICENSE ] && cp LICENSE "$BUNDLE/Contents/Resources/"

# --------------------------------------------------------------- assinatura

if [ "$assinar" -eq 1 ]; then
	IDENTIDADE="$(resolver_identidade)"
	passo "Assinando com '$IDENTIDADE'"
	# `-o runtime` liga o Hardened Runtime, que a notarização exige.
	codesign --force --deep --timestamp -o runtime \
		--entitlements app/Resources/Aguardente.entitlements \
		--sign "$IDENTIDADE" "$BUNDLE"

	passo "Verificando a assinatura"
	codesign --verify --strict --verbose=2 "$BUNDLE"
	# Antes da notarização o Gatekeeper ainda recusa: é o esperado, e é
	# exatamente isso que o passo seguinte resolve.
	spctl --assess --type execute --verbose=4 "$BUNDLE" || true
else
	passo "Pulando a assinatura (--no-sign)"
fi

# -------------------------------------------------------------- notarização

if [ "$notarizar" -eq 1 ]; then
	[ "$assinar" -eq 1 ] || { echo "--notarize exige assinatura" >&2; exit 2; }

	passo "Enviando para notarização"
	ZIP="$DESTINO/$NOME.zip"
	rm -f "$ZIP"
	ditto -c -k --keepParent "$BUNDLE" "$ZIP"
	xcrun notarytool submit "$ZIP" --keychain-profile "$PERFIL_NOTARIZACAO" --wait

	passo "Grampeando o tíquete ao bundle"
	xcrun stapler staple "$BUNDLE"
	xcrun stapler validate "$BUNDLE"

	passo "Reempacotando com o tíquete"
	rm -f "$ZIP"
	ditto -c -k --keepParent "$BUNDLE" "$ZIP"
	echo "distribuível: $ZIP"
fi

passo "Pronto: $BUNDLE"
