import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from core.defense_engine import DefenseEngine


def test_gaussian_defense_returns_image():
    engine = DefenseEngine()
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = engine.gaussian_defense(img, sigma=1.5)
    assert isinstance(result, Image.Image)
    assert result.size == (224, 224)


def test_jpeg_defense_returns_image():
    engine = DefenseEngine()
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = engine.jpeg_defense(img, quality=50)
    assert isinstance(result, Image.Image)
    assert result.size == (224, 224)


def test_apply_defense_dispatcher():
    engine = DefenseEngine()
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = engine.apply_defense(img, method="gaussian", sigma=2.0)
    assert isinstance(result, Image.Image)
