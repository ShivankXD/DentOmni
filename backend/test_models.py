"""Quick smoke test for ResNet50 and Faster R-CNN models."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.INFO)

from backend.model_loader import ResNet50Classifier, FasterRCNNModel
from PIL import Image
import numpy as np

print("=== Testing ResNet50 Classifier ===")
clf = ResNet50Classifier()
clf.load()

black = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
r1 = clf.predict(black)
print(f"Black image: caries={r1['caries']['confidence']:.3f}  lesion={r1['periapical_lesion']['confidence']:.3f}")

white = Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8))
r2 = clf.predict(white)
print(f"White image: caries={r2['caries']['confidence']:.3f}  lesion={r2['periapical_lesion']['confidence']:.3f}")

rng_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
rng = Image.fromarray(rng_arr)
r3 = clf.predict(rng)
print(f"Random img:  caries={r3['caries']['confidence']:.3f}  lesion={r3['periapical_lesion']['confidence']:.3f}")

print()
print("=== Testing Faster R-CNN ===")
det = FasterRCNNModel()
det.load()
r4 = det.predict(rng)
print(f"FRCNN caries: {r4['caries']}")
print(f"FRCNN lesion: {r4['periapical_lesion']}")
print(f"FRCNN detections: {len(r4['bounding_boxes'])} boxes")
print()
print("SUCCESS: Both models loaded and predicted without errors!")
