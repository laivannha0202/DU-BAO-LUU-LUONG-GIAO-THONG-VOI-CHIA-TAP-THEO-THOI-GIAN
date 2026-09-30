"""
app/__init__.py

Ứng dụng FastAPI phục vụ mô hình Ridge đã đóng băng.

Nguyên tắc bất di bất dịch của lớp serving:
- CHỈ load artifact đã đóng băng (`models/ridge_pipeline.joblib`,
  `models/model_metadata.json`, ...). KHÔNG train, KHÔNG tune alpha.
- Mọi feature được tạo bằng CHUNG hàm với lúc huấn luyện
  (`src.features.build_features`) => không có train-serving skew.
- KHÔNG đọc target từ dataset để dự báo.
- KHÔNG fit encoder / imputer / scaler mới: tất cả đã nằm sẵn trong pipeline
  và được fit trên TRAIN ở `src/train.py`.
"""

__all__ = ["config", "registry", "schemas", "serving", "main"]
