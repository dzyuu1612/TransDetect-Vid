"""TransDetect-Vid: phát hiện đối tượng trong video dựa trên phép biến đổi.

Package gom hai nhánh giải quyết cùng bài toán:
- Nhánh truyền thống (phép biến đổi thủ công): preprocessing -> classical_detector
  -> optical_flow.
- Nhánh học sâu (phép biến đổi học được): yolo_detector.
"""

from . import preprocessing
from . import classical_detector
from . import optical_flow
from . import yolo_detector
from . import visualization
from . import config

__all__ = [
    "preprocessing",
    "classical_detector",
    "optical_flow",
    "yolo_detector",
    "visualization",
    "config",
]

__version__ = "0.1.0"
