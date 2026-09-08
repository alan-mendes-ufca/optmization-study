# Comparação de inferência no Raspberry Pi 5

Medição realizada via SSH em 08/09/2026, Raspberry Pi 5 Model B Rev 1.1, aarch64, Docker `edgeai_lab`, TensorFlow 2.13.0, CPU. Uma thread por runtime, cinco aquecimentos e 100 inferências por modelo, divididas em cinco blocos de 20 com ordem alternada.

Imagem: `imagem-benchmark.jpg`, copiada da imagem de teste já existente no projeto de visão computacional do Raspberry. A imagem mostra duas pessoas de terno. Entrada RGB 224×224, normalizada em [-1, 1]. Os tempos incluem a chamada e a obtenção da saída; excluem download, carregamento dos modelos, leitura, pré-processamento e decodificação das classes. TFLite inclui cópia da entrada. O baseline usa a chamada eager do roteiro, sem `tf.function`; o ganho mede também a diferença entre runtimes, não apenas representação numérica.

Temperatura antes/depois: 49,4 °C / 52,1 °C. `get_throttled`: `0x0` antes e depois.

| Modelo | Tempo médio (s) | Variação contra baseline | Classe principal | Observações |
|---|---:|---:|---|---|
| TensorFlow baseline | 0.144517 | 0.00% | suit (terno) | Referência; execução eager |
| TFLite FP32 | 0.022711 | -84.29% | suit (terno) | 13.34 MiB; confiança 71.30% |
| TFLite quantizado | 0.021160 | -85.36% | suit (terno) | 3.61 MiB; confiança 70.56% |

Variação = (tempo / baseline − 1) × 100. Valores negativos indicam redução.

1. **Menor tempo:** TFLite quantizado, 0.021160 s, aproximadamente 6.83 vezes mais rápido que o baseline.
2. **Ganho da quantização:** em relação ao TFLite FP32, reduziu o tempo em 6.83% e o tamanho do arquivo em 72.91%. O ganho adicional de latência é modesto; a economia de armazenamento é expressiva. A maior parte da redução contra o baseline já ocorre com TFLite FP32. Não foi realizado teste formal de significância estatística nem medida de RAM ou consumo energético.
3. **Qualidade:** a classe principal permaneceu `suit` e o top-3 manteve a ordem `suit`, `academic_gown`, `bow_tie`. A confiança principal foi de 71,30% (baseline), 71,30% (FP32) e 70,56% (quantizado), queda de aproximadamente 0,74 ponto percentual. Não houve mudança perceptível da classificação nesta imagem. Uma única imagem não permite concluir que a acurácia geral foi preservada; isso exige um conjunto de validação rotulado.
4. **Configuração embarcada:** entre as três configurações testadas neste Raspberry, o TFLite quantizado oferece o melhor compromisso entre latência e tamanho, condicionado à validação de qualidade com imagens da aplicação. É quantização dinâmica, com pesos quantizados e entrada/saída FP32, não INT8 integral.

Desvio-padrão dos tempos: baseline 11.565 ms; FP32 0.034 ms; quantizado 0.055 ms. O baseline apresentou um máximo de 258.74 ms, preservado na média e nos dados brutos.

## Reprodução

Na raiz do projeto no Raspberry:

```bash
docker run --rm -e TF_CPP_MIN_LOG_LEVEL=2 -v "$PWD:/app" edgeai_lab python scripts/benchmark_comparativo.py resultados/imagem-benchmark.jpg
```

O script e a imagem estão preservados no projeto. `benchmark.json` contém as 100 medições de cada versão, top-3, tamanhos e hashes SHA-256 dos modelos TFLite e da imagem.
