"""Response supervision derived from the checkpoint's actual chat template."""
from .errors import AguardenteError


def assistant_supervision(tokenizer, messages, template_kwargs=None):
    if not getattr(tokenizer, "chat_template", None):
        raise AguardenteError("supervisão de respostas exige chat_template do tokenizer")
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        raise AguardenteError("messages deve ser uma lista de mensagens")
    kwargs = dict(template_kwargs or {})
    if {"tokenize", "add_generation_prompt", "return_tensors"} & kwargs.keys():
        raise AguardenteError("template_kwargs não pode alterar tokenização ou geração de prefixos")

    def render(items, generation=False):
        return tokenizer.apply_chat_template(items, tokenize=False,
                                              add_generation_prompt=generation, **kwargs)

    def encode(text):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    text = render(messages).strip()
    ids = encode(text)
    mask = [0] * len(ids)
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if not index:
            raise AguardenteError("resposta sem um prompt anterior")
        prefix = encode(render(messages[:index], generation=True).lstrip())
        through = encode(render(messages[:index + 1]).strip())
        # Some templates rewrite prior turns or inject reasoning prefixes. A
        # guessed boundary could silently teach prompt tokens as answers.
        if ids[:len(through)] != through or through[:len(prefix)] != prefix or len(prefix) >= len(through):
            raise AguardenteError("template não permite delimitar respostas com prefixos estáveis; forneça um adaptador específico")
        mask[len(prefix):len(through)] = [1] * (len(through) - len(prefix))
    if not any(mask[1:]):
        raise AguardenteError("conversa sem tokens de resposta supervisionáveis")
    return text, mask
