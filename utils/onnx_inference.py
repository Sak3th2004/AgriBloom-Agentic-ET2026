"""
ONNX Vision Inference Engine with GPU Acceleration
Supports NVIDIA RTX 4060 via CUDAExecutionProvider
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalization (used by ViT)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_optimal_providers() -> list[str]:
    """Get optimal ONNX execution providers with CUDA priority."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()

        providers = []
        # Prioritize GPU providers
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB limit for RTX 4060
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }))
        if "TensorRTExecutionProvider" in available:
            providers.insert(0, ("TensorRTExecutionProvider", {
                "device_id": 0,
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                "trt_fp16_enable": True,
            }))
        # Always add CPU as fallback
        providers.append("CPUExecutionProvider")

        return providers
    except ImportError:
        return ["CPUExecutionProvider"]


class ONNXVisionEngine:
    """
    Production-grade ONNX inference engine for crop disease detection.
    Automatically uses GPU if available (CUDA/TensorRT).
    """

    def __init__(
        self,
        model_path: str = "models/vision/vit_base_patch16_224.onnx",
        force_cpu: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.force_cpu = force_cpu
        self.session: Optional["ort.InferenceSession"] = None
        self.providers_used: list[str] = []
        self._initialized = False

        self._lazy_init()

    def _lazy_init(self) -> None:
        """Lazy initialization of ONNX session."""
        if self._initialized:
            return

        import onnxruntime as ort

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.model_path}")

        # Select providers
        if self.force_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers = get_optimal_providers()

        # Create session with optimal settings
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )
            self.providers_used = self.session.get_providers()
            logger.info(f"ONNX session initialized with providers: {self.providers_used}")
        except Exception as e:
            logger.warning(f"Failed to init with GPU providers: {e}. Falling back to CPU.")
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.providers_used = ["CPUExecutionProvider"]

        self._initialized = True

    @staticmethod
    def preprocess(image: Image.Image, size: int = 224) -> np.ndarray:
        """
        Preprocess image for ViT model.
        Uses ImageNet normalization matching HuggingFace ViTImageProcessor.
        """
        # Convert to RGB and resize
        rgb = image.convert("RGB").resize((size, size), Image.BILINEAR)

        # Convert to numpy and normalize
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD

        # CHW format with batch dimension
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)

        return arr.astype(np.float32)

    def infer(self, image: Image.Image) -> np.ndarray:
        """
        Run inference on a single image.

        Args:
            image: PIL Image to classify

        Returns:
            Logits array (num_classes,)
        """
        if not self._initialized:
            self._lazy_init()

        tensor = self.preprocess(image)

        # Get input name from model
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        outputs = self.session.run([output_name], {input_name: tensor})
        return outputs[0][0]

    def infer_batch(self, images: list[Image.Image]) -> np.ndarray:
        """
        Run batch inference for multiple images.

        Args:
            images: List of PIL Images

        Returns:
            Logits array (batch_size, num_classes)
        """
        if not self._initialized:
            self._lazy_init()

        # Stack preprocessed images
        tensors = np.concatenate([self.preprocess(img) for img in images], axis=0)

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        outputs = self.session.run([output_name], {input_name: tensors})
        return outputs[0]

    def get_prediction(self, image: Image.Image, class_names: list[str]) -> dict:
        """
        Get prediction with class name and confidence.

        Args:
            image: PIL Image
            class_names: List of class names

        Returns:
            Dict with label, confidence, and provider info
        """
        logits = self.infer(image)

        # Softmax
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        # Map to class name safely
        if idx < len(class_names):
            label = class_names[idx]
        else:
            label = class_names[idx % len(class_names)]

        return {
            "label": label,
            "confidence": confidence,
            "class_index": idx,
            "provider": self.providers_used[0] if self.providers_used else "unknown",
            "source": "onnx_gpu" if "CUDA" in str(self.providers_used) else "onnx_cpu",
        }

    @property
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is active."""
        return any("CUDA" in p or "TensorRT" in p for p in self.providers_used)


# Module-level singleton for reuse
_GLOBAL_ENGINE: Optional[ONNXVisionEngine] = None


def get_engine(model_path: str = "models/vision/vit_base_patch16_224.onnx") -> ONNXVisionEngine:
    """Get or create global ONNX engine instance."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        try:
            _GLOBAL_ENGINE = ONNXVisionEngine(model_path)
        except FileNotFoundError:
            logger.warning(f"ONNX model not found at {model_path}")
            raise
    return _GLOBAL_ENGINE
