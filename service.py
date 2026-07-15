from __future__ import annotations

import base64
import hmac
import logging
import os
from http import HTTPStatus
from pathlib import Path
from typing import Any, Literal, Union

import bentoml
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/harrier"))
ONNX_MODEL_FILE = os.getenv("ONNX_MODEL_FILE", "onnx/model.onnx")
MODEL_FILE = MODEL_DIR / ONNX_MODEL_FILE
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "harrier-0.6b-int8")
MODEL_ALIASES = {
    name.strip()
    for name in os.getenv("MODEL_ALIASES", "").split(",")
    if name.strip()
}
EXPECTED_DIMENSIONS = int(os.getenv("EXPECTED_DIMENSIONS", "1024"))

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))

ORT_INTRA_OP_THREADS = int(os.getenv("ORT_INTRA_OP_THREADS", "0"))
ORT_INTER_OP_THREADS = int(os.getenv("ORT_INTER_OP_THREADS", "1"))
ORT_DISABLE_SPINNING = os.getenv("ORT_DISABLE_SPINNING", "0") == "1"
WARMUP = os.getenv("WARMUP", "1") == "1"

API_WORKERS = int(os.getenv("API_WORKERS", "1"))
API_THREADS = int(os.getenv("API_THREADS", "1"))
MODEL_WORKERS = int(os.getenv("MODEL_WORKERS", "1"))
MODEL_THREADS = int(os.getenv("MODEL_THREADS", "1"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
MODEL_MAX_CONCURRENCY = int(os.getenv("MODEL_MAX_CONCURRENCY", "8"))

ADAPTIVE_BATCHING = os.getenv("ADAPTIVE_BATCHING", "1") == "1"
ADAPTIVE_MAX_BATCH_SIZE = int(os.getenv("ADAPTIVE_MAX_BATCH_SIZE", "8"))
ADAPTIVE_MAX_LATENCY_MS = int(os.getenv("ADAPTIVE_MAX_LATENCY_MS", "10"))


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    input: Union[str, list[str]]
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: Union[list[float], str]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: Usage


_ONNX_DTYPES: dict[str, np.dtype[Any]] = {
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(bool)": np.dtype(np.bool_),
}


@bentoml.service(
    name="harrier_inference",
    workers=MODEL_WORKERS,
    threads=MODEL_THREADS,
    traffic={
        "timeout": REQUEST_TIMEOUT,
        "max_concurrency": MODEL_MAX_CONCURRENCY,
    },
)
class HarrierInferenceService:
    def __init__(self) -> None:
        if not MODEL_FILE.is_file():
            raise FileNotFoundError(
                f"Model not found at {MODEL_FILE}; check the model directory mount"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            use_fast=True,
        )
        self.tokenizer.padding_side = "right"

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = ORT_INTRA_OP_THREADS
        session_options.inter_op_num_threads = ORT_INTER_OP_THREADS
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True

        if ORT_DISABLE_SPINNING:
            session_options.add_session_config_entry(
                "session.intra_op.allow_spinning", "0"
            )
            session_options.add_session_config_entry(
                "session.inter_op.allow_spinning", "0"
            )

        self.session = ort.InferenceSession(
            str(MODEL_FILE),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.session_inputs = {item.name: item for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]

        logger.info(
            "Harrier ONNX model loaded: providers=%s inputs=%s outputs=%s "
            "max_length=%d max_batch_size=%d intra_op_threads=%d "
            "inter_op_threads=%d",
            self.session.get_providers(),
            [
                (item.name, item.type, item.shape)
                for item in self.session.get_inputs()
            ],
            [
                (item.name, item.type, item.shape)
                for item in self.session.get_outputs()
            ],
            MAX_LENGTH,
            MAX_BATCH_SIZE,
            ORT_INTRA_OP_THREADS,
            ORT_INTER_OP_THREADS,
        )

        if WARMUP:
            embeddings, _ = self._embed(["warmup"])
            if embeddings.shape != (1, EXPECTED_DIMENSIONS):
                raise RuntimeError(
                    f"Warmup returned unexpected shape {embeddings.shape}"
                )

    @staticmethod
    def _position_ids(attention_mask: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        positions = np.cumsum(attention_mask, axis=1, dtype=np.int64) - 1
        return np.maximum(positions, 0)

    def _create_feeds(
        self,
        encoded: dict[str, np.ndarray[Any, Any]],
    ) -> dict[str, np.ndarray[Any, Any]]:
        input_ids = np.asarray(encoded["input_ids"])
        attention_mask = np.asarray(encoded["attention_mask"])

        candidates: dict[str, np.ndarray[Any, Any]] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": self._position_ids(attention_mask),
            "token_type_ids": np.zeros_like(input_ids),
        }

        feeds: dict[str, np.ndarray[Any, Any]] = {}
        for name, metadata in self.session_inputs.items():
            if name not in candidates:
                raise RuntimeError(
                    f"Unsupported ONNX input {name!r}; "
                    f"known tokenizer inputs are {sorted(candidates)}"
                )

            dtype = _ONNX_DTYPES.get(metadata.type)
            if dtype is None:
                raise RuntimeError(
                    f"Unsupported ONNX input dtype {metadata.type!r} for {name!r}"
                )
            feeds[name] = np.ascontiguousarray(candidates[name], dtype=dtype)

        return feeds

    def _extract_embeddings(
        self,
        outputs: dict[str, np.ndarray[Any, Any]],
        attention_mask: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        preferred_names = (
            "sentence_embedding",
            "embedding",
            "embeddings",
        )

        for name in preferred_names:
            if name not in outputs:
                continue
            candidate = np.asarray(outputs[name])
            if candidate.ndim == 3 and candidate.shape[1] == 1:
                candidate = candidate[:, 0, :]
            if candidate.ndim == 2:
                return candidate

        for candidate in outputs.values():
            candidate = np.asarray(candidate)
            if (
                candidate.ndim == 2
                and candidate.shape[-1] == EXPECTED_DIMENSIONS
            ):
                return candidate

        hidden_state = None
        for candidate in outputs.values():
            candidate = np.asarray(candidate)
            if (
                candidate.ndim == 3
                and candidate.shape[-1] == EXPECTED_DIMENSIONS
            ):
                hidden_state = candidate
                break

        if hidden_state is None:
            shapes = {name: np.asarray(value).shape for name, value in outputs.items()}
            raise RuntimeError(
                f"Cannot identify embedding output; outputs: {shapes}"
            )

        sequence_positions = np.arange(attention_mask.shape[1])[None, :]
        last_indices = np.where(
            attention_mask.astype(bool),
            sequence_positions,
            -1,
        ).max(axis=1)

        if np.any(last_indices < 0):
            raise RuntimeError("Cannot pool an input with no non-padding token")

        return hidden_state[np.arange(hidden_state.shape[0]), last_indices]

    def _embed(
        self,
        texts: list[str],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        encoded_batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_tensors="np",
        )
        encoded = {
            key: np.asarray(value)
            for key, value in encoded_batch.items()
        }

        attention_mask = np.asarray(encoded["attention_mask"])
        feeds = self._create_feeds(encoded)

        raw_outputs = self.session.run(None, feeds)
        outputs = dict(zip(self.output_names, raw_outputs, strict=True))
        embeddings = self._extract_embeddings(outputs, attention_mask)

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape != (len(texts), EXPECTED_DIMENSIONS):
            raise RuntimeError(
                "Unexpected embedding shape: "
                f"expected {(len(texts), EXPECTED_DIMENSIONS)}, "
                f"got {embeddings.shape}"
            )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

        token_counts = attention_mask.sum(axis=1, dtype=np.int64)
        return embeddings, token_counts

    @bentoml.api(
        batchable=ADAPTIVE_BATCHING,
        max_batch_size=ADAPTIVE_MAX_BATCH_SIZE,
        max_latency_ms=ADAPTIVE_MAX_LATENCY_MS,
    )
    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings, token_counts = self._embed(texts)
        return np.column_stack((embeddings, token_counts.astype(np.float32)))


@bentoml.service(
    name="harrier_embedding",
    workers=API_WORKERS,
    threads=API_THREADS,
    traffic={
        "timeout": REQUEST_TIMEOUT,
        "max_concurrency": MAX_CONCURRENCY,
    },
)
class HarrierEmbeddingService:
    inference = bentoml.depends(HarrierInferenceService)

    def __init__(self) -> None:
        self.api_key = os.getenv("API_KEY", "")

    def _authorize(self, ctx: bentoml.Context) -> None:
        if not self.api_key:
            return

        supplied = ctx.request.headers.get("authorization", "")
        expected = f"Bearer {self.api_key}"
        if not hmac.compare_digest(supplied, expected):
            raise bentoml.exceptions.BentoMLException(
                "Unauthorized",
                error_code=HTTPStatus.UNAUTHORIZED,
            )

    @staticmethod
    def _normalize_texts(value: Union[str, list[str]]) -> list[str]:
        texts = [value] if isinstance(value, str) else value

        if not texts:
            raise bentoml.exceptions.BadInput("Input must not be empty")
        if len(texts) > MAX_BATCH_SIZE:
            raise bentoml.exceptions.BadInput(
                f"Batch size {len(texts)} exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}"
            )
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise bentoml.exceptions.BadInput(
                "Every input item must be a non-empty string"
            )
        return texts

    @staticmethod
    def _serialize_embedding(
        embedding: np.ndarray[Any, Any],
        encoding_format: Literal["float", "base64"],
    ) -> list[float] | str:
        if encoding_format == "float":
            return embedding.tolist()

        raw = np.asarray(embedding, dtype=np.dtype("<f4")).tobytes()
        return base64.b64encode(raw).decode("ascii")

    @bentoml.api(
        route="/v1/embeddings",
        input_spec=EmbeddingRequest,
        output_spec=EmbeddingResponse,
    )
    async def embeddings(
        self,
        ctx: bentoml.Context,
        **params: Any,
    ) -> EmbeddingResponse:
        self._authorize(ctx)

        request = EmbeddingRequest(**params)
        allowed_models = {SERVED_MODEL_NAME, *MODEL_ALIASES}
        if request.model not in allowed_models:
            raise bentoml.exceptions.BadInput(
                f"Unknown model {request.model!r}; use one of "
                f"{sorted(allowed_models)!r}"
            )
        texts = self._normalize_texts(request.input)

        encoded = np.asarray(await self.inference.to_async.encode(texts))
        if encoded.shape != (len(texts), EXPECTED_DIMENSIONS + 1):
            raise RuntimeError(f"Unexpected internal output shape {encoded.shape}")

        embeddings = encoded[:, :EXPECTED_DIMENSIONS]
        token_count = int(encoded[:, EXPECTED_DIMENSIONS].sum())

        return EmbeddingResponse(
            data=[
                EmbeddingItem(
                    embedding=self._serialize_embedding(
                        embedding,
                        request.encoding_format,
                    ),
                    index=index,
                )
                for index, embedding in enumerate(embeddings)
            ],
            model=SERVED_MODEL_NAME,
            usage=Usage(
                prompt_tokens=token_count,
                total_tokens=token_count,
            ),
        )
