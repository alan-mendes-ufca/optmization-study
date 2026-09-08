"""Compara MobileNetV2 e os TFLite, com entrada local e uma thread."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from comum import carregar_imagem, preparar_mobilenet

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("imagem")
parser.add_argument("--saida", default="resultados/benchmark.json")
args = parser.parse_args()
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
entrada = preparar_mobilenet(carregar_imagem(args.imagem))
baseline = tf.keras.applications.MobileNetV2(weights="imagenet")
runners = {"TensorFlow baseline": lambda: baseline(entrada, training=False).numpy()}
metadata = {}
for nome, arquivo in [("TFLite FP32", "modelo/mobilenet_fp32.tflite"),
                      ("TFLite quantizado", "modelo/mobilenet_int8.tflite")]:
    interpreter = tf.lite.Interpreter(model_path=arquivo, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    assert inp["dtype"] == np.float32 and out["dtype"] == np.float32
    def run(i=interpreter, ix=inp["index"], ox=out["index"]):
        i.set_tensor(ix, entrada)
        i.invoke()
        return i.get_tensor(ox)
    runners[nome] = run
    data = Path(arquivo).read_bytes()
    metadata[nome] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "input_dtype": str(inp["dtype"]), "output_dtype": str(out["dtype"])}
tempos = {nome: [] for nome in runners}
preds = {}
for nome, run in runners.items():
    for _ in range(5):
        preds[nome] = run()
# Alterna a ordem entre blocos para reduzir o efeito da ordem de execução.
nomes = list(runners)
for bloco in range(5):
    for nome in nomes[bloco % 3:] + nomes[:bloco % 3]:
        for _ in range(20):
            inicio = time.perf_counter()
            preds[nome] = runners[nome]()
            tempos[nome].append(time.perf_counter() - inicio)
result = {"tensorflow": tf.__version__, "machine": platform.machine(),
          "threads": 1, "warmups": 5, "repetitions": 100,
          "image_sha256": hashlib.sha256(Path(args.imagem).read_bytes()).hexdigest(),
          "timing": "entrada ja pre-processada; inclui chamada e copia da saida; baseline eager",
          "models": {}}
for nome in runners:
    t = tempos[nome]
    top = tf.keras.applications.mobilenet_v2.decode_predictions(preds[nome], top=3)[0]
    result["models"][nome] = {**metadata.get(nome, {}), "mean_s": float(np.mean(t)),
        "std_s": float(np.std(t, ddof=1)), "min_s": min(t), "max_s": max(t),
        "times_s": t, "top3": [[code, label, float(p)] for code, label, p in top],
        "max_absolute_output_delta_vs_baseline": float(np.max(np.abs(preds[nome]-preds[nomes[0]])))}
Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
Path(args.saida).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({**result, "models": {n: {k: v for k, v in m.items() if k != "times_s"} for n, m in result["models"].items()}}, indent=2))
