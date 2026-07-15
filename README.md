# harrier-bentoml-onnxruntime

## Model

- [Harrier OSS v1 0.6B ONNX](https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX)
- [Harrier OSS v1 270M ONNX](https://huggingface.co/onnx-community/harrier-oss-v1-270m-ONNX)

These are ONNX Community conversions of the Microsoft Harrier checkpoints. For
CPU inference, download `onnx/model_quantized.onnx` and its matching
`onnx/model_quantized.onnx_data`, together with the configuration and tokenizer
files, into `models/harrier`:

```text
models/harrier/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── onnx/
    ├── model_quantized.onnx
    └── model_quantized.onnx_data
```

## Run

```bash
docker run -d \
  --name harrier \
  --restart unless-stopped \
  --env-file .env \
  --cpus 8 \
  --memory 6g \
  -p 127.0.0.1:7997:3000 \
  --mount type=bind,src="$(pwd)/models/harrier",dst=/models/harrier,readonly \
  ghcr.io/zhousiru/harrier-bentoml-onnxruntime:latest
```

## Request

```bash
curl http://127.0.0.1:7997/v1/embeddings \
  -H "Authorization: Bearer replace-with-a-long-random-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "harrier-0.6b-int8",
    "input": "Instruct: Retrieve relevant passages\nQuery: How do I run embeddings on a CPU?"
  }'
```

The server does not modify `input`. Format Harrier queries as
`Instruct: <instruction>\nQuery: <text>` and send documents as plain text.
`encoding_format` may be `float` or `base64`.

## Environment

| Variable                                             | Default             | Purpose                                                    |
| ---------------------------------------------------- | ------------------- | ---------------------------------------------------------- |
| `API_KEY`                                            | required            | Bearer token; an empty value disables authentication.      |
| `MODEL_DIR`                                          | `/models/harrier`   | Model directory inside the container.                      |
| `ONNX_MODEL_FILE`                                    | `onnx/model.onnx`   | ONNX file relative to `MODEL_DIR`.                         |
| `SERVED_MODEL_NAME`                                  | `harrier-0.6b-int8` | Model name accepted and returned by the API.               |
| `MODEL_ALIASES`                                      | empty               | Additional accepted model names, separated by commas.      |
| `EXPECTED_DIMENSIONS`                                | `1024`              | Output width; Harrier 270M uses `640`.                     |
| `MAX_LENGTH`                                         | `512`               | Maximum tokenized length; longer input is truncated.       |
| `MAX_BATCH_SIZE`                                     | `8`                 | Maximum strings in one request.                            |
| `ADAPTIVE_BATCHING`                                  | `1`                 | Enable or disable cross-request batching.                  |
| `ADAPTIVE_MAX_BATCH_SIZE`, `ADAPTIVE_MAX_LATENCY_MS` | `8`, `10`           | Merged batch limit and wait time in milliseconds.          |
| `ORT_INTRA_OP_THREADS`, `ORT_INTER_OP_THREADS`       | `8`, `1`            | ONNX Runtime CPU thread pools.                             |
| `ORT_DISABLE_SPINNING`                               | `1`                 | Reduce contention on a virtualized host.                   |
| `API_WORKERS`, `MODEL_WORKERS`                       | `1`, `1`            | Worker counts; each model worker loads another model copy. |
| `API_THREADS`, `MODEL_THREADS`                       | `1`, `1`            | BentoML threads per worker.                                |
| `MAX_CONCURRENCY`, `MODEL_MAX_CONCURRENCY`           | `8`, `8`            | Public and model-service concurrency limits.               |
| `REQUEST_TIMEOUT`                                    | `120`               | Request timeout in seconds.                                |
| `WARMUP`                                             | `1`                 | Run one inference when the model worker starts.            |
