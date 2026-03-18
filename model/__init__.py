# CoreVision v2 - Model Package
from model.pipeline import VehicleRecognitionPipeline
from model.video_processor import VideoProcessor
from model.detector import VehiclePlateDetector
from model.car_classifier import CarClassifier
from model.plate_ocr import PlateOCR
from model.country_identifier import CountryIdentifier

__all__ = [
    'VehicleRecognitionPipeline',
    'VideoProcessor',
    'VehiclePlateDetector',
    'CarClassifier',
    'PlateOCR',
    'CountryIdentifier',
]
