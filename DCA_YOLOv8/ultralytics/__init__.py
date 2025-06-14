# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.5"

#from ICA_Detection.external.DCA_YOLOv8.DCA_YOLOv8.ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import RTDETR, SAM, YOLO
from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS
from ultralytics.utils import ASSETS, SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download
import ultralytics.nn as nn

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
    "nn",
)
