"""Configuração comum da suíte.

Sem `torch`, os testes de poda, destilação e desserialização são pulados — e
são justamente as áreas de maior risco. Uma suíte verde nessas condições diz
menos do que parece, então a ausência é anunciada no cabeçalho e no resumo em
vez de ficar registrada só como contagem de skips.
"""

import importlib.util

TORCH_AUSENTE = importlib.util.find_spec("torch") is None

AVISO = ("As etapas de poda, destilação e desserialização de logits não foram "
         "exercidas. Rode a suíte completa com: "
         "uv run --with pytest --with torch --with transformers pytest")


def pytest_report_header(config):
    if TORCH_AUSENTE:
        return "aguardente: torch ausente — cobertura reduzida nas etapas do pipeline"
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not TORCH_AUSENTE:
        return
    terminalreporter.write_sep("=", "cobertura reduzida: torch ausente", yellow=True)
    terminalreporter.write_line(AVISO)
