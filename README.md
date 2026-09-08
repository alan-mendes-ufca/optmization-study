# Otimização de modelos de visão computacional

Projeto prático da Aula 8 para converter e comparar a MobileNetV2 original com versões TensorFlow Lite FP32 e quantizada. O ambiente foi preparado para Python 3.10 e TensorFlow 2.13 em Docker, conforme o enunciado em `docs/Aula 8.docx`.

## Estrutura

```text
.
├── Dockerfile
├── modelo/                       # modelos gerados localmente
├── resultados/                   # resultados experimentais
├── scripts/
│   ├── inferencia_url.py         # baseline TensorFlow
│   ├── conversao_tflite.py       # conversão TFLite FP32
│   ├── quantizacao_tflite.py     # quantização dinâmica pós-treinamento
│   └── inferencia_tflite_url.py  # inferência TFLite
└── requirements.txt
```

## Executar com Docker

No Raspberry Pi 5, use um sistema operacional de 64 bits. O comando
`uname -m` deve retornar `aarch64`.

Na raiz do projeto:

```bash
docker build --pull -t edgeai_lab .
docker run --rm -it -v "$(pwd):/app" edgeai_lab
```

Para confirmar a arquitetura e a versão do TensorFlow instaladas:

```bash
docker run --rm edgeai_lab python -c \
  'import platform, tensorflow as tf; print(platform.machine(), tf.__version__)'
```

No Raspberry Pi 5, o resultado esperado começa com `aarch64 2.13.0`. A imagem
usa CPU; este projeto não configura aceleração por GPU, Hailo ou Coral.

Dentro do contêiner, gere os modelos:

```bash
python scripts/conversao_tflite.py
python scripts/quantizacao_tflite.py
```

Use a mesma imagem em todas as medições. A origem pode ser uma URL pública ou um arquivo local:

```bash
python scripts/inferencia_url.py "URL_OU_ARQUIVO"
python scripts/inferencia_tflite_url.py modelo/mobilenet_fp32.tflite "URL_OU_ARQUIVO"
python scripts/inferencia_tflite_url.py modelo/mobilenet_int8.tflite "URL_OU_ARQUIVO"
```

Se a origem for omitida, cada script a solicitará no terminal. Os scripts fazem uma inferência de aquecimento e medem várias repetições com `time.perf_counter()`, reduzindo a influência da inicialização no resultado.

## Registrar a comparação

Anote o tempo médio informado por cada execução:

| Modelo | Tempo médio (s) | Variação contra baseline | Classe principal | Observações |
|---|---:|---:|---|---|
| TensorFlow baseline |  | 0% |  |  |
| TFLite FP32 |  |  |  |  |
| TFLite quantizado |  |  |  |  |

A variação percentual pode ser calculada por:

```text
((tempo_do_modelo - tempo_baseline) / tempo_baseline) * 100
```

Valores negativos indicam redução do tempo. Compare também o tamanho dos arquivos em `modelo/` e as três classes previstas.

## Observação sobre INT8

O procedimento da aula usa `Optimize.DEFAULT` sem um conjunto representativo. Isso produz quantização dinâmica: principalmente os pesos são armazenados em 8 bits, enquanto entrada e saída permanecem em ponto flutuante. O nome `mobilenet_int8.tflite` foi mantido por compatibilidade com o roteiro, mas não representa quantização inteira completa. Para INT8 integral seria necessário fornecer imagens representativas ao conversor e configurar os tipos de entrada e saída.
