"""Deterministic keyword-based skill extractor.

No LLM involved — uses regex word-boundary matching against a curated
skill vocabulary. Returns a lowercase normalised set of detected skills.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Curated skill vocabulary
# Multi-word entries are matched as phrases; single-word as whole words.
# ---------------------------------------------------------------------------

_SKILLS: list[str] = [
    # --- Languages ---
    "python", "c++", "c#", "java", "javascript", "typescript", "scala",
    "rust", "go", "golang", "r", "matlab", "julia", "kotlin", "swift",
    # --- ML / DL Frameworks ---
    "pytorch", "tensorflow", "keras", "jax", "paddle",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    # --- Computer Vision ---
    "opencv", "pillow", "albumentations", "timm",
    "yolo", "yolov5", "yolov8", "yolov11",
    "detectron2", "mmdetection", "torchvision",
    "object detection", "instance segmentation", "semantic segmentation",
    "pose estimation", "keypoint estimation", "object tracking",
    "image classification", "2d reconstruction", "3d reconstruction",
    "depth estimation", "optical flow",
    # --- NLP / LLM ---
    "transformers", "hugging face", "huggingface", "bert", "gpt",
    "llm", "large language model", "rag", "retrieval augmented generation",
    "langchain", "llamaindex", "llama index",
    "embeddings", "vector search", "semantic search",
    "text generation", "summarization", "question answering",
    # --- Agentic AI ---
    "multi-agent", "tool use", "function calling", "react agent",
    "langgraph", "autogen", "crewai", "agentic",
    # --- MLOps / Experimentation ---
    "mlflow", "wandb", "weights and biases", "dvc", "neptune",
    "airflow", "kubeflow", "sagemaker", "vertex ai",
    # --- Optimisation / Inference ---
    "onnx", "openvino", "tensorrt", "tflite", "coreml",
    "quantization", "pruning", "distillation",
    "latency-aware inference", "edge deployment",
    # --- Containerisation / DevOps ---
    "docker", "kubernetes", "k8s", "helm", "terraform",
    "ci/cd", "github actions", "gitlab ci", "jenkins",
    "microservices", "rest api", "grpc", "graphql",
    # --- Cloud ---
    "aws", "azure", "gcp", "google cloud",
    "ec2", "s3", "lambda", "cloud functions",
    # --- Vector / Databases ---
    "qdrant", "pinecone", "weaviate", "chroma", "milvus", "faiss",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    # --- Data Engineering ---
    "spark", "kafka", "dbt", "pandas", "numpy", "pyspark",
    "data pipeline", "etl", "feature engineering",
    # --- Web / API ---
    "fastapi", "flask", "django", "uvicorn",
    # --- Research domains ---
    "tomography", "computational geometry", "signal processing",
    "inverse problems", "computed tomography",
    "ultrasound", "x-ray", "medical imaging",
    # --- General AI / ML ---
    "computer vision", "machine learning", "deep learning",
    "neural network", "cnn", "rnn", "lstm", "transformer",
    "attention mechanism", "transfer learning", "fine-tuning",
    "reinforcement learning", "self-supervised learning",
    # --- Soft / Leadership ---
    "leadership", "mentoring", "research", "collaboration",
    "technical planning", "system design",
]

# Pre-compile patterns: multi-word first so longer matches win
_MULTI_WORD = [(s, re.compile(r'\b' + re.escape(s) + r'\b', re.IGNORECASE))
               for s in _SKILLS if ' ' in s or '-' in s or '/' in s]
_SINGLE_WORD = [(s, re.compile(r'\b' + re.escape(s) + r'\b', re.IGNORECASE))
                for s in _SKILLS if ' ' not in s and '-' not in s and '/' not in s]


def extract_skills(text: str) -> set[str]:
    """Return a set of lowercase normalised skill strings found in *text*."""
    found: set[str] = set()

    # Collapse newlines and excess whitespace so multi-word skills
    # (e.g. "machine learning") are not broken by PDF extraction artefacts
    # like "machine\n \nlearning".
    text = " ".join(text.split())

    # Replace multi-word matches first so they are not broken by single-word pass
    masked = text
    for skill, pattern in _MULTI_WORD:
        if pattern.search(masked):
            found.add(skill.lower())

    for skill, pattern in _SINGLE_WORD:
        if pattern.search(masked):
            found.add(skill.lower())

    return found
