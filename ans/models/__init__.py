"""
ANS Model Definitions
=====================
Original VGG-16 and Tiny-YOLO models with partitionable forward pass.
"""

from ans.models.vgg16 import vgg16, Vgg16
from ans.models.tiny_yolo import tinyYolo

__all__ = ["vgg16", "Vgg16", "tinyYolo"]
