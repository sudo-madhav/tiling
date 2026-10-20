"""Frozen encoders for with_encoder. Nothing here downloads weights."""

from .base import Encoder, EncoderUnavailable, FeatureMap
from .registry import build_encoder, build_stack

__all__ = ["Encoder", "EncoderUnavailable", "FeatureMap", "build_encoder", "build_stack"]
