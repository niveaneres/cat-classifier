from src.services.model import ModelHandler
from unittest.mock import patch, MagicMock
import numpy as np
import unittest

class TestModelHandler(unittest.TestCase):
    @patch("os.getenv")
    @patch("cv2.dnn.readNetFromCaffe")
    def test_modelhandler_init(self, mock_readNet, mock_getenv):
        mock_getenv.side_effect = lambda k: "dummy_path"
        handler = ModelHandler()
        assert handler.prototxt == "dummy_path"
        assert handler.model == "dummy_path"
        mock_readNet.assert_called_once_with("dummy_path", "dummy_path")

    @patch("os.getenv", return_value="dummy_path")
    @patch("cv2.dnn.readNetFromCaffe")
    @patch("cv2.dnn.blobFromImage")
    @patch("cv2.resize")
    @patch("src.services.model.transform")
    def test_inference_cat(self,
        mock_transform, mock_resize, mock_blob
    ):
        handler = ModelHandler()
        mock_net = MagicMock()
        handler.net = mock_net
        mock_transform.return_value = np.zeros((500, 500, 3), dtype=np.uint8)
        mock_resize.return_value = np.zeros((416, 416, 3), dtype=np.uint8)
        mock_blob.return_value = "blob"
    
        detections = np.zeros((1, 1, 1, 7))
        detections[0, 0, 0, 2] = 0.9  
        detections[0, 0, 0, 1] = 8   
        mock_net.forward.return_value = detections
        result = handler.inference(np.zeros((500, 500, 3), dtype=np.uint8))
        assert result == "cat"

    @patch("os.getenv", return_value="dummy_path")
    @patch("cv2.dnn.readNetFromCaffe")
    @patch("cv2.dnn.blobFromImage")
    @patch("cv2.resize")
    @patch("src.services.model.transform")
    def test_inference_not_cat(self,
        mock_transform, mock_resize, mock_blob
    ):
        handler = ModelHandler()
        mock_net = MagicMock()
        handler.net = mock_net
        mock_transform.return_value = np.zeros((500, 500, 3), dtype=np.uint8)
        mock_resize.return_value = np.zeros((416, 416, 3), dtype=np.uint8)
        mock_blob.return_value = "blob"
    
        detections = np.zeros((1, 1, 1, 7))
        detections[0, 0, 0, 2] = 0.7  
        detections[0, 0, 0, 1] = 3    
        mock_net.forward.return_value = detections
        result = handler.inference(np.zeros((500, 500, 3), dtype=np.uint8))
        assert result == "it's not a cat"


if __name__ == "__main__":
    unittest.main()
