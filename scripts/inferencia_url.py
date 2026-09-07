"""Executa a inferencia baseline com a MobileNetV2 no TensorFlow."""

from __future__ import annotations

import argparse
import time

import tensorflow as tf

from comum import carregar_imagem, exibir_predicoes, preparar_mobilenet, solicitar_origem


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origem", nargs="?", help="URL HTTP(S) ou caminho de uma imagem")
    parser.add_argument("--repeticoes", type=int, default=5, help="numero de inferencias medidas")
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    if args.repeticoes < 1:
        raise SystemExit("--repeticoes deve ser maior que zero")

    origem = solicitar_origem(args.origem)
    entrada = preparar_mobilenet(carregar_imagem(origem))

    print("Carregando MobileNetV2...")
    modelo = tf.keras.applications.MobileNetV2(weights="imagenet")

    # Aquecimento evita contabilizar inicializacoes internas no tempo medido.
    modelo(entrada, training=False)
    tempos = []
    saida = None
    for _ in range(args.repeticoes):
        inicio = time.perf_counter()
        saida = modelo(entrada, training=False).numpy()
        tempos.append(time.perf_counter() - inicio)

    assert saida is not None
    predicoes = tf.keras.applications.mobilenet_v2.decode_predictions(saida, top=3)[0]
    exibir_predicoes(predicoes)
    print(f"\nTempo medio ({args.repeticoes} execucoes): {sum(tempos) / len(tempos):.4f} s")
    print(f"Menor tempo: {min(tempos):.4f} s")


if __name__ == "__main__":
    main()
