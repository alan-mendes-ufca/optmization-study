"""Executa inferencia por URL ou arquivo local com um modelo TensorFlow Lite."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from comum import carregar_imagem, exibir_predicoes, preparar_mobilenet, solicitar_origem


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modelo", type=Path, help="arquivo .tflite")
    parser.add_argument("origem", nargs="?", help="URL HTTP(S) ou caminho de uma imagem")
    parser.add_argument("--repeticoes", type=int, default=10, help="numero de inferencias medidas")
    parser.add_argument("--threads", type=int, default=1, help="threads usadas pelo interpretador")
    return parser.parse_args()


def quantizar(dados: np.ndarray, detalhe: dict) -> np.ndarray:
    dtype = detalhe["dtype"]
    if np.issubdtype(dtype, np.floating):
        return dados.astype(dtype)
    escala, zero = detalhe["quantization"]
    if escala == 0:
        raise RuntimeError("Modelo possui entrada inteira sem parametros de quantizacao validos")
    limites = np.iinfo(dtype)
    return np.clip(np.rint(dados / escala + zero), limites.min, limites.max).astype(dtype)


def desquantizar(dados: np.ndarray, detalhe: dict) -> np.ndarray:
    if np.issubdtype(dados.dtype, np.floating):
        return dados.astype(np.float32)
    escala, zero = detalhe["quantization"]
    if escala == 0:
        raise RuntimeError("Modelo possui saida inteira sem parametros de quantizacao validos")
    return (dados.astype(np.float32) - zero) * escala


def main() -> None:
    args = argumentos()
    if not args.modelo.is_file():
        raise SystemExit(f"Modelo nao encontrado: {args.modelo}")
    if args.repeticoes < 1 or args.threads < 1:
        raise SystemExit("--repeticoes e --threads devem ser maiores que zero")

    entrada_float = preparar_mobilenet(carregar_imagem(solicitar_origem(args.origem)))
    interpretador = tf.lite.Interpreter(model_path=str(args.modelo), num_threads=args.threads)
    interpretador.allocate_tensors()
    detalhe_entrada = interpretador.get_input_details()[0]
    detalhe_saida = interpretador.get_output_details()[0]
    entrada = quantizar(entrada_float, detalhe_entrada)

    interpretador.set_tensor(detalhe_entrada["index"], entrada)
    interpretador.invoke()  # aquecimento

    tempos = []
    for _ in range(args.repeticoes):
        interpretador.set_tensor(detalhe_entrada["index"], entrada)
        inicio = time.perf_counter()
        interpretador.invoke()
        tempos.append(time.perf_counter() - inicio)

    saida = desquantizar(interpretador.get_tensor(detalhe_saida["index"]), detalhe_saida)
    predicoes = tf.keras.applications.mobilenet_v2.decode_predictions(saida, top=3)[0]
    print(f"Modelo: {args.modelo}")
    exibir_predicoes(predicoes)
    print(f"\nTempo medio ({args.repeticoes} execucoes): {sum(tempos) / len(tempos):.4f} s")
    print(f"Menor tempo: {min(tempos):.4f} s")


if __name__ == "__main__":
    main()
