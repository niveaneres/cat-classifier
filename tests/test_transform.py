from src.services.utils import transform
import unittest
from PIL import Image
import numpy as np
import io


class TestTransform(unittest.TestCase):
    def test_transform_valid_image(self):
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        
        result = transform(img_bytes)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_transform_invalid_bytes(self):
        invalid_bytes = b"notanimage"
        with self.assertRaises(Exception):
            transform(invalid_bytes)

if __name__ == "__main__":
    unittest.main()