"""Aplica quantizacao dinamica pos-treinamento aos pesos da MobileNetV2."""

from pathlib import Path

import tensorflow as tf


DESTINO = Path("modelo/mobilenet_int8.tflite")


def main() -> None:
    print("Carregando modelo...")
    modelo = tf.keras.applications.MobileNetV2(weights="imagenet")

    print("Aplicando quantizacao dinamica pos-treinamento...")
    conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conversor.optimizations = [tf.lite.Optimize.DEFAULT]
    modelo_tflite = conversor.convert()

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_bytes(modelo_tflite)
    print(f"Modelo quantizado salvo em {DESTINO} ({DESTINO.stat().st_size / 1024 / 1024:.2f} MiB)")
    print("Nota: os pesos sao quantizados; entrada e saida continuam em ponto flutuante.")


if __name__ == "__main__":
    main()
