#!/usr/bin/env bash
#
# Instalador do aguardente para macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/SEU-USUARIO/aguardente/main/install.sh | bash
#
# Resolve tudo: interpretador Python correto, gerenciador de pacotes,
# acelerador de download, e o proprio programa. Reexecutar e seguro.

set -euo pipefail

PYTHON_VERSION="3.12"
REPO_URL="${AGUARDENTE_REPO:-https://github.com/SEU-USUARIO/aguardente}"

bold=$'\033[1m'; dim=$'\033[2m'; red=$'\033[31m'; green=$'\033[32m'
yellow=$'\033[33m'; blue=$'\033[34m'; reset=$'\033[0m'
[ -t 1 ] || { bold=""; dim=""; red=""; green=""; yellow=""; blue=""; reset=""; }

step()  { printf "\n%s==>%s %s%s%s\n" "$blue" "$reset" "$bold" "$1" "$reset"; }
ok()    { printf "    %s✓%s %s\n" "$green" "$reset" "$1"; }
warn()  { printf "    %s!%s %s\n" "$yellow" "$reset" "$1"; }
die()   { printf "\n%serro:%s %s\n" "$red" "$reset" "$1" >&2; [ $# -gt 1 ] && printf "  %s\n" "$2" >&2; exit 1; }

printf "%s\n" "${bold}aguardente${reset} ${dim}— instalacao${reset}"

# ---------------------------------------------------------------- 1. sistema
step "Verificando o sistema"

[ "$(uname -s)" = "Darwin" ] || die "este instalador e para macOS" \
  "O Core AI da Apple nao existe em outras plataformas."

[ "$(uname -m)" = "arm64" ] || die "e necessario um Mac com Apple Silicon" \
  "O runtime Core AI nao publica binarios para Intel."
ok "macOS em Apple Silicon"

macos_major=$(sw_vers -productVersion | cut -d. -f1)
if [ "$macos_major" -lt 27 ]; then
  warn "macOS $(sw_vers -productVersion) — a conversao final exige 27 ou superior"
  warn "As etapas de poda e destilacao funcionam mesmo assim."
else
  ok "macOS $(sw_vers -productVersion)"
fi

# ---------------------------------------------------------------- 2. homebrew
step "Verificando o Homebrew"
if command -v brew >/dev/null 2>&1; then
  ok "ja instalado"
else
  warn "ausente — instalando"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  for p in /opt/homebrew/bin /usr/local/bin; do
    [ -x "$p/brew" ] && eval "$("$p/brew" shellenv)"
  done
  command -v brew >/dev/null 2>&1 || die "Homebrew instalou mas nao entrou no PATH" \
    "Abra um terminal novo e rode este script de novo."
  ok "instalado"
fi

# ---------------------------------------------------------------- 3. aria2
step "Verificando o aria2"
if command -v aria2c >/dev/null 2>&1; then
  ok "$(aria2c --version | head -1)"
else
  warn "ausente — instalando (downloads retomaveis)"
  brew install aria2
  ok "instalado"
fi

# ---------------------------------------------------------------- 4. uv
step "Verificando o uv"
if command -v uv >/dev/null 2>&1; then
  ok "$(uv --version)"
else
  warn "ausente — instalando"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv instalou mas nao entrou no PATH" \
    "Adicione ~/.local/bin ao PATH e rode de novo."
  ok "instalado"
fi

# ---------------------------------------------------------------- 5. python
step "Preparando o Python ${PYTHON_VERSION}"
# O python3 do sistema costuma ser 3.13/3.14, fora da faixa que o stack aceita.
uv python install "$PYTHON_VERSION" >/dev/null 2>&1 || true
ok "interpretador ${PYTHON_VERSION} disponivel"

# ---------------------------------------------------------------- 6. programa
step "Instalando o aguardente"
if [ -f "pyproject.toml" ] && grep -q 'name = "aguardente"' pyproject.toml 2>/dev/null; then
  ok "instalando a partir deste diretorio"
  uv tool install --python "$PYTHON_VERSION" --force --with-editable . "aguardente[pipeline] @ ."
else
  ok "instalando a partir de ${REPO_URL}"
  uv tool install --python "$PYTHON_VERSION" --force "aguardente[pipeline] @ git+${REPO_URL}"
fi

export PATH="$HOME/.local/bin:$PATH"
command -v aguardente >/dev/null 2>&1 || die "o comando 'aguardente' nao entrou no PATH" \
  "Rode: export PATH=\"\$HOME/.local/bin:\$PATH\" e adicione ao seu ~/.zshrc"
ok "$(aguardente --version)"

# ---------------------------------------------------------------- 7. conferir
step "Conferindo o ambiente"
aguardente doctor || true

cat <<FIM

${bold}Pronto.${reset}

  Comece por aqui — nao baixa nada, so mostra o plano:
    ${bold}aguardente plan Qwen/Qwen3-4B${reset}

  Ensaio completo com um modelo pequeno (poucos minutos):
    ${bold}aguardente run HuggingFaceTB/SmolLM2-135M-Instruct -o ensaio --target-params 90e6${reset}

  Guia detalhado: ${dim}usage.md${reset}
FIM
