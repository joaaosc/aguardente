"""Testes do módulo de adaptadores LoRA para camadas nn.Linear."""

from __future__ import annotations

import math
import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from aguardente.distill.adapters import LoRAError, LoRALinear, attach_lora, merge_lora


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int = 16, hidden_dim: int = 32, out_dim: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class NestedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(16, 16, bias=True),
            nn.Linear(16, 16, bias=False),
        ])
        self.head = nn.Linear(16, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)


class TiedWeightsModel(nn.Module):
    def __init__(self, vocab_size: int = 20, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        # Amarra os pesos do lm_head com a tabela de embeddings
        self.lm_head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = torch.relu(self.hidden(h))
        return self.lm_head(h)


class OutputEmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(16, 16)
        self.output_layer = nn.Linear(16, 10)

    def get_output_embeddings(self) -> nn.Linear:
        return self.output_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(torch.relu(self.fc(x)))


class NoLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


def test_explicit_tied_target_rejected_without_freezing():
    model = TiedWeightsModel()
    with pytest.raises(LoRAError, match="compartilha"):
        attach_lora(model, targets=("lm_head",))
    assert all(p.requires_grad for p in model.parameters())


def test_failed_merge_preserves_all_base_weights():
    model = SimpleMLP()
    attach_lora(model, rank=2)
    before = model.fc1.base_layer.weight.detach().clone()
    with torch.no_grad():
        model.fc1.lora_B.fill_(1)
        model.fc2.lora_B.fill_(float("nan"))
    with pytest.raises(LoRAError):
        merge_lora(model)
    assert torch.equal(model.fc1.base_layer.weight, before)


def test_dtype_move_does_not_round_adapter_weights():
    model = SimpleMLP()
    attach_lora(model, rank=2)
    before = model.fc1.lora_A.detach().clone()
    model.half()
    assert torch.equal(model.fc1.lora_A, before)


def test_forward_identity_before_training() -> None:
    """Antes de qualquer treino, a inicialização B=0 garante que a saída é idêntica à base."""
    torch.manual_seed(42)
    model = SimpleMLP()
    x = torch.randn(4, 16)
    expected = model(x)

    info = attach_lora(model, rank=4, alpha=8.0)
    assert info["modules"] == ["fc1", "fc2"]
    assert info["trainable_params"] > 0
    assert info["frozen_params"] > 0
    assert isinstance(model.fc1, LoRALinear)
    assert isinstance(model.fc2, LoRALinear)

    actual = model(x)
    assert torch.equal(actual, expected)


def test_optimizer_step_changes_adapters_while_base_unchanged() -> None:
    """Passo de otimizador treina os adaptadores LoRA mantendo pesos base estritamente intocados."""
    torch.manual_seed(42)
    model = SimpleMLP()

    w1_orig = model.fc1.weight.clone()
    b1_orig = model.fc1.bias.clone()
    w2_orig = model.fc2.weight.clone()

    attach_lora(model, rank=4, alpha=8.0)

    # Base deve estar congelada
    assert not model.fc1.base_layer.weight.requires_grad
    assert not model.fc1.base_layer.bias.requires_grad
    assert not model.fc2.base_layer.weight.requires_grad

    # Adaptadores devem estar treináveis
    assert model.fc1.lora_A.requires_grad
    assert model.fc1.lora_B.requires_grad
    assert model.fc2.lora_A.requires_grad
    assert model.fc2.lora_B.requires_grad

    a1_orig = model.fc1.lora_A.clone()
    b1_lora_orig = model.fc1.lora_B.clone()

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    x = torch.randn(4, 16)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    # Camada base inalterada
    assert torch.equal(model.fc1.base_layer.weight, w1_orig)
    assert torch.equal(model.fc1.base_layer.bias, b1_orig)
    assert torch.equal(model.fc2.base_layer.weight, w2_orig)

    # Adaptadores foram modificados
    assert not torch.equal(model.fc1.lora_B, b1_lora_orig)


def test_merge_parity_including_nonzero_delta_and_bias() -> None:
    """Paridade exata entre saída do LoRA com delta/viés não nulos e saída da camada fundida."""
    torch.manual_seed(123)
    model = SimpleMLP()
    attach_lora(model, rank=4, alpha=8.0)

    # Torna o delta não-nulo alterando lora_B
    with torch.no_grad():
        model.fc1.lora_B.normal_(0, 0.5)
        model.fc2.lora_B.normal_(0, 0.5)

    x = torch.randn(3, 16)
    y_before = model(x)

    merged_count = merge_lora(model)
    assert merged_count == 2
    assert isinstance(model.fc1, nn.Linear)
    assert not isinstance(model.fc1, LoRALinear)
    assert isinstance(model.fc2, nn.Linear)
    assert not isinstance(model.fc2, LoRALinear)

    y_after = model(x)
    assert torch.allclose(y_before, y_after, atol=1e-5)

    # Chamada repetida é no-op
    repeat_count = merge_lora(model)
    assert repeat_count == 0
    assert torch.allclose(model(x), y_after)


def test_merge_row_chunking_large_layer() -> None:
    """Fusão em blocos de até 128 linhas funciona em camadas com mais de 128 saídas."""
    torch.manual_seed(42)
    in_dim, out_dim = 32, 300
    linear = nn.Linear(in_dim, out_dim, bias=True)
    model = nn.Sequential(linear)

    attach_lora(model, rank=8, alpha=16.0)
    with torch.no_grad():
        model[0].lora_B.normal_(0, 0.1)

    x = torch.randn(2, in_dim)
    y_before = model(x)

    merged = merge_lora(model)
    assert merged == 1
    assert isinstance(model[0], nn.Linear)
    assert not isinstance(model[0], LoRALinear)

    y_after = model(x)
    assert torch.allclose(y_before, y_after, atol=1e-5)


def test_tied_output_weight_exclusion() -> None:
    """Módulos que compartilham peso base com outro módulo são excluídos quando targets=()."""
    model = TiedWeightsModel()
    info = attach_lora(model)

    assert info["modules"] == ["hidden"]
    assert isinstance(model.hidden, LoRALinear)
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, LoRALinear)
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_output_embeddings_exclusion() -> None:
    """model.get_output_embeddings() é excluído de alvos vazios."""
    model = OutputEmbeddingModel()
    info = attach_lora(model)

    assert info["modules"] == ["fc"]
    assert isinstance(model.fc, LoRALinear)
    assert isinstance(model.output_layer, nn.Linear)
    assert not isinstance(model.output_layer, LoRALinear)


def test_unmatched_and_invalid_inputs_leave_requires_grad_unchanged() -> None:
    """Validações que falham antes da mutação deixam requires_grad intacto."""
    model = SimpleMLP()
    initial_grads = [p.requires_grad for p in model.parameters()]
    assert all(initial_grads)

    # 1. Alvo que não corresponde a nenhum módulo
    with pytest.raises(ValueError):
        attach_lora(model, targets=("nonexistent_submodule",))
    assert [p.requires_grad for p in model.parameters()] == initial_grads
    assert isinstance(model.fc1, nn.Linear)

    # 2. Alvo parcial inválido
    with pytest.raises(ValueError):
        attach_lora(model, targets=("fc1", "invalid_layer"))
    assert [p.requires_grad for p in model.parameters()] == initial_grads
    assert isinstance(model.fc1, nn.Linear)

    # 3. Rank inválido: booleano
    with pytest.raises((ValueError, TypeError)):
        attach_lora(model, rank=True)
    assert [p.requires_grad for p in model.parameters()] == initial_grads

    with pytest.raises((ValueError, TypeError)):
        attach_lora(model, rank=False)
    assert [p.requires_grad for p in model.parameters()] == initial_grads

    # 4. Rank não positivo ou tipo incorreto
    with pytest.raises(ValueError):
        attach_lora(model, rank=0)
    with pytest.raises(ValueError):
        attach_lora(model, rank=-5)
    with pytest.raises((ValueError, TypeError)):
        attach_lora(model, rank=4.5)  # type: ignore
    assert [p.requires_grad for p in model.parameters()] == initial_grads

    # 5. Alpha inválido: não positivo, infinito, nan, booleano
    with pytest.raises(ValueError):
        attach_lora(model, alpha=0.0)
    with pytest.raises(ValueError):
        attach_lora(model, alpha=-16.0)
    with pytest.raises(ValueError):
        attach_lora(model, alpha=float("inf"))
    with pytest.raises(ValueError):
        attach_lora(model, alpha=float("nan"))
    with pytest.raises((ValueError, TypeError)):
        attach_lora(model, alpha=True)
    assert [p.requires_grad for p in model.parameters()] == initial_grads


def test_already_adapted_model_error() -> None:
    """Tentativa de anexar LoRA a um modelo já adaptado lança erro sem corrompê-lo."""
    model = SimpleMLP()
    attach_lora(model, rank=4)
    with pytest.raises(ValueError):
        attach_lora(model, rank=4)


def test_no_eligible_modules_error() -> None:
    """Modelo sem camadas nn.Linear elegíveis lança erro claro."""
    model = NoLinearModel()
    with pytest.raises(ValueError):
        attach_lora(model)


def test_target_suffix_matching() -> None:
    """Targets não vazios aceitam sufixos de nomes de módulos."""
    model = NestedModel()
    # Nome dos submódulos: layers.0, layers.1, head
    info = attach_lora(model, targets=("head", "layers.0"))
    assert sorted(info["modules"]) == ["head", "layers.0"]
    assert isinstance(model.head, LoRALinear)
    assert isinstance(model.layers[0], LoRALinear)
    assert isinstance(model.layers[1], nn.Linear)
    assert not isinstance(model.layers[1], LoRALinear)


def test_non_finite_merge_detection() -> None:
    """Fusão detecta valores infinitos ou NaN e lança exceção clara."""
    model = SimpleMLP()
    attach_lora(model, rank=4)
    with torch.no_grad():
        model.fc1.lora_A[0, 0] = float("nan")
        model.fc1.lora_B[0, 0] = 1.0

    with pytest.raises(ValueError):
        merge_lora(model)


def test_output_shape_and_dtype_preservation() -> None:
    """LoRALinear preserva formato multidimensional e tipos de ponto flutuante."""
    model = SimpleMLP(16, 32, 8)
    attach_lora(model, rank=4)

    # Batches com mais de 2 dimensões
    x3d = torch.randn(2, 5, 16)
    out3d = model(x3d)
    assert out3d.shape == (2, 5, 8)

    # Entrada unidimensional
    x1d = torch.randn(16)
    out1d = model(x1d)
    assert out1d.shape == (8,)

    # Preservação de FP16 se suportado na CPU/dispositivo
    model_fp16 = SimpleMLP(16, 32, 8).half()
    attach_lora(model_fp16, rank=4)
    x_fp16 = torch.randn(2, 16, dtype=torch.float16)
    out_fp16 = model_fp16(x_fp16)
    assert out_fp16.dtype == torch.float16
    assert out_fp16.shape == (2, 8)
    merge_lora(model_fp16)
    assert model_fp16.fc1.weight.dtype == torch.float16
