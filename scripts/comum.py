"""Funcoes compartilhadas pelos scripts de inferencia."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError


TAMANHO_ENTRADA = (224, 224)


def carregar_imagem(origem: str, timeout: float = 20.0) -> Image.Image:
    """Carrega uma imagem de uma URL HTTP(S) ou de um arquivo local."""
    try:
        if origem.startswith(("http://", "https://")):
            resposta = requests.get(origem, timeout=timeout)
            resposta.raise_for_status()
            imagem = Image.open(BytesIO(resposta.content))
        else:
            imagem = Image.open(Path(origem).expanduser())
        return imagem.convert("RGB")
    except (requests.RequestException, OSError, UnidentifiedImageError) as erro:
        raise RuntimeError(f"Nao foi possivel carregar a imagem: {erro}") from erro


def preparar_mobilenet(imagem: Image.Image) -> np.ndarray:
    """Aplica o pre-processamento esperado pela MobileNetV2: pixels em [-1, 1]."""
    imagem = imagem.resize(TAMANHO_ENTRADA, Image.Resampling.BILINEAR)
    dados = np.asarray(imagem, dtype=np.float32)
    dados = dados / 127.5 - 1.0
    return np.expand_dims(dados, axis=0)


def solicitar_origem(valor: str | None) -> str:
    return valor or input("Digite a URL ou o caminho da imagem: ").strip()


def exibir_predicoes(predicoes: list[tuple[str, str, float]]) -> None:
    print("\nResultado:")
    for _codigo, classe, confianca in predicoes:
        print(f"{classe} - {confianca * 100:.2f}%")
