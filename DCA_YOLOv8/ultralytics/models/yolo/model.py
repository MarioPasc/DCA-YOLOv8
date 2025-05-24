# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel

import logging  # Add this import
from collections import defaultdict # Add this import if not already present for _callbacks

# Get a logger instance
logger = logging.getLogger(__name__) # Add this line


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (str, Path): Path to the model file to load or create.
            task (str | None): Task type for the model.
            verbose (bool): Specifies if the YOLO model should be verbose or not. Defaults to False.
        """
        super().__init__(model=model, task=task, verbose=verbose) # Call the parent's __init__
        logger.info("Initializing DCA-YOLOv8 implementation.") # Add this log message

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""

        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }
