"""Converte a MobileNetV2 para TensorFlow Lite em FP32."""

from pathlib import Path

import tensorflow as tf


DESTINO = Path("modelo/mobilenet_fp32.tflite")


def main() -> None:
    print("Carregando modelo...")
    modelo = tf.keras.applications.MobileNetV2(weights="imagenet")

    print("Convertendo para TFLite FP32...")
    conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    modelo_tflite = conversor.convert()

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_bytes(modelo_tflite)
    print(f"Modelo FP32 salvo em {DESTINO} ({DESTINO.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
