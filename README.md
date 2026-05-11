# Dự Án Phân Tích Học Tập OULAD

## Mô tả

Dự án này là một hệ thống phân tích dữ liệu học tập sử dụng bộ dữ liệu OULAD (Open University Learning Analytics Dataset). Dự án bao gồm pipeline học máy để xử lý dữ liệu, huấn luyện mô hình, và triển khai API để dự đoán kết quả học tập của sinh viên.

## Tính năng chính

- **Pipeline Học Máy**: Xử lý dữ liệu thô, kỹ thuật đặc trưng, và huấn luyện mô hình.
- **Theo dõi Thử nghiệm**: Sử dụng MLflow để theo dõi các thử nghiệm và mô hình.
- **API Phục vụ**: Triển khai mô hình qua API REST sử dụng FastAPI.
- **Container hóa**: Sử dụng Docker và Docker Compose để dễ dàng triển khai.
- **Quản lý Dữ liệu**: Sử dụng DVC để quản lý phiên bản dữ liệu.

## Công nghệ sử dụng

- **Ngôn ngữ**: Python 3.9+
- **Framework**: FastAPI, Scikit-learn, Pandas, NumPy
- **Công cụ**: MLflow, DVC, Docker, Docker Compose
- **Thư viện ML**: Optuna (tuning hyperparameters), LightGBM/XGBoost (mô hình)

## Cài đặt

### Yêu cầu hệ thống

- Docker và Docker Compose
- Python 3.9+ (nếu chạy không dùng Docker)

### Chạy với Docker (Khuyến nghị)

1. Clone repository:
   ```bash
   git clone <repository-url>
   cd project
   ```

2. Chạy các container:
   ```bash
   docker compose up --build
   ```

   Điều này sẽ khởi động:
   - API server trên port 8000
   - MLflow UI trên port 5000

### Cài đặt thủ công

1. Tạo virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Windows: venv\Scripts\activate
   ```

2. Cài đặt dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements.api.txt
   ```

3. Chạy pipeline:
   ```bash
   python main.py
   ```

4. Chạy API:
   ```bash
   cd api
   uvicorn app:app --reload
   ```

## Cách sử dụng

### API Endpoints

- `GET /`: Thông tin về API
- `POST /predict`: Dự đoán kết quả học tập dựa trên dữ liệu đầu vào

Ví dụ sử dụng API:

```python
import requests

data = {
    "feature1": value1,
    "feature2": value2,
    # ... các đặc trưng khác
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### MLflow UI

Truy cập http://localhost:5000 để xem các thử nghiệm và mô hình đã lưu.

## Cấu trúc dự án

```
.
├── api/                    # Mã nguồn API
│   └── app.py
├── configs/                # Cấu hình dự án
├── data/                   # Dữ liệu
│   ├── raw/                # Dữ liệu thô
│   ├── processed/          # Dữ liệu đã xử lý
│   └── features/           # Đặc trưng đã trích xuất
├── logs/                   # Log files
├── models/                 # Mô hình đã huấn luyện
├── src/                    # Mã nguồn chính
│   ├── ingestion/          # Nhập dữ liệu
│   ├── pipeline/           # Pipeline xử lý
│   ├── processing/         # Xử lý dữ liệu
│   ├── serving/            # Phục vụ mô hình
│   ├── training/           # Huấn luyện mô hình
│   └── utils/              # Tiện ích
├── tests/                  # Test cases
├── docker-compose.yml      # Cấu hình Docker Compose
├── Dockerfile.api          # Dockerfile cho API
├── Dockerfile.mlflow       # Dockerfile cho MLflow
├── main.py                 # Script chính
├── requirements.txt        # Dependencies chính
└── requirements.api.txt    # Dependencies cho API
```

## Đóng góp

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request


## Liên hệ

- Tác giả: [Tên của bạn]
- Email: [Email của bạn]
- GitHub: [Link GitHub]

## Tài liệu tham khảo

- [OULAD Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)