#!/usr/bin/env bash
#
# Script de instalação do aguardente para macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/joaaosc/aguardente/main/install.sh | bash
#
# Configura o interpretador Python, dependências do sistema e instala o pacote.

set -euo pipefail

PYTHON_VERSION="3.12"
REPO_URL="${AGUARDENTE_REPO:-https://github.com/joaaosc/aguardente}"

bold=$'\033[1m'; dim=$'\033[2m'; red=$'\033[31m'; green=$'\033[32m'
yellow=$'\033[33m'; blue=$'\033[34m'; reset=$'\033[0m'
[ -t 1 ] || { bold=""; dim=""; red=""; green=""; yellow=""; blue=""; reset=""; }

step()  { printf "\n%s==>%s %s%s%s\n" "$blue" "$reset" "$bold" "$1" "$reset"; }
ok()    { printf "    %s✓%s %s\n" "$green" "$reset" "$1"; }
warn()  { printf "    %s!%s %s\n" "$yellow" "$reset" "$1"; }
die()   { printf "\n%serro:%s %s\n" "$red" "$reset" "$1" >&2; [ $# -gt 1 ] && printf "  %s\n" "$2" >&2; exit 1; }

printf "%s\n" "${bold}aguardente${reset} ${dim}— instalação${reset}"

# ---------------------------------------------------------------- 1. sistema
step "Verificando o sistema"

[ "$(uname -s)" = "Darwin" ] || die "este instalador é exclusivo para macOS" \
  "O framework Core AI está disponível apenas no macOS."

[ "$(uname -m)" = "arm64" ] || die "é necessário um Mac com Apple Silicon" \
  "O runtime Core AI requer arquitetura arm64."
ok "macOS em Apple Silicon"

macos_major=$(sw_vers -productVersion | cut -d. -f1)
if [ "$macos_major" -lt 27 ]; then
  warn "macOS $(sw_vers -productVersion) detectado — exportação para .aimodel requer macOS 27+"
  warn "As etapas de poda e destilação continuam operacionais."
else
  ok "macOS $(sw_vers -productVersion)"
fi

# ---------------------------------------------------------------- 2. homebrew
step "Verificando o Homebrew"
if command -v brew >/dev/null 2>&1; then
  ok "Homebrew instalado"
else
  warn "Homebrew não encontrado — instalando"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  for p in /opt/homebrew/bin /usr/local/bin; do
    [ -x "$p/brew" ] && eval "$("$p/brew" shellenv)"
  done
  command -v brew >/dev/null 2>&1 || die "Homebrew instalado mas não localizado no PATH" \
    "Abra uma nova sessão de terminal e execute novamente."
  ok "Homebrew instalado"
fi

# ---------------------------------------------------------------- 3. aria2
step "Verificando o aria2"
if command -v aria2c >/dev/null 2>&1; then
  ok "$(aria2c --version | head -1)"
else
  warn "aria2c não encontrado — instalando via brew"
  brew install aria2
  ok "aria2 instalado"
fi

# ---------------------------------------------------------------- 4. uv
step "Verificando o uv"
if command -v uv >/dev/null 2>&1; then
  ok "$(uv --version)"
else
  warn "uv não encontrado — instalando"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv instalado mas não localizado no PATH" \
    "Adicione ~/.local/bin ao PATH e execute novamente."
  ok "uv instalado"
fi

# ---------------------------------------------------------------- 5. python
step "Configurando Python ${PYTHON_VERSION}"
uv python install "$PYTHON_VERSION" >/dev/null 2>&1 || true
ok "interpretador Python ${PYTHON_VERSION} configurado"

# ---------------------------------------------------------------- 6. programa
step "Instalando o aguardente"
if [ -f "pyproject.toml" ] && grep -q 'name = "aguardente"' pyproject.toml 2>/dev/null; then
  ok "instalando a partir do repositório local"
  uv tool install --python "$PYTHON_VERSION" --force --with-editable . "aguardente[pipeline] @ ."
else
  ok "instalando a partir de ${REPO_URL}"
  uv tool install --python "$PYTHON_VERSION" --force "aguardente[pipeline] @ git+${REPO_URL}"
fi

export PATH="$HOME/.local/bin:$PATH"
command -v aguardente >/dev/null 2>&1 || die "comando 'aguardente' não encontrado no PATH" \
  "Adicione export PATH=\"\$HOME/.local/bin:\$PATH\" ao seu ~/.zshrc"
ok "$(aguardente --version)"

# ---------------------------------------------------------------- 7. conferir
step "Executando diagnóstico inicial"
aguardente doctor || true

cat <<FIM

${bold}Instalação concluída.${reset}

  Inspecione o modelo antes de baixar pesos:
    ${bold}aguardente plan Qwen/Qwen3-4B${reset}

  Ou execute um ensaio rápido com modelo leve:
    ${bold}aguardente run HuggingFaceTB/SmolLM2-135M-Instruct -o ensaio --target-params 90e6${reset}

  Consulte ${dim}usage.md${reset} para opções detalhadas.
FIM
