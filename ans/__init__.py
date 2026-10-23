"""
Autodidactic Neurosurgeon (ANS) - Original Codebase
===================================================
Direct import of the original repository: https://github.com/letian-zhang/ANS
Includes the original muLinUCB controller, VGG-16, Tiny-YOLO, socket communication,
and key frame detection modules without alterations.
"""

from ans.muLinUCB import muLinUCB, fillThetaContext, getCx
from ans.partition_profiles import vgg_info, yolo_info

__all__ = ["muLinUCB", "fillThetaContext", "getCx", "vgg_info", "yolo_info"]
