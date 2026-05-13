# Student Learning Analytics Pipeline

[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docs.docker.com/compose/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-orange)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.0+-red)](https://airflow.apache.org/)

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Tính năng chính](#-tính-năng-chính)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt và chạy](#-cài-đặt-và-chạy)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Đóng góp](#-đóng-góp)
- [Giấy phép](#-giấy-phép)

## 🎯 Tổng quan

Dự án này xây dựng một pipeline MLOps toàn diện cho phân tích học tập dựa trên bộ dữ liệu OULAD (Open University Learning Analytics Dataset). Hệ thống sử dụng machine learning để dự đoán hiệu suất học tập và xác định học sinh có nguy cơ rớt môn, giúp giáo viên và nhà trường can thiệp kịp thời.

### Mục tiêu chính:
- **Dự đoán hiệu suất**: Xây dựng mô hình dự đoán điểm số cuối kỳ của học sinh
- **Phát hiện rủi ro**: Xác định học sinh có nguy cơ không hoàn thành khóa học
- **Tự động hóa pipeline**: Sử dụng Airflow để tự động hóa quy trình từ xử lý dữ liệu đến triển khai mô hình
- **Theo dõi thí nghiệm**: MLflow quản lý vòng đời mô hình và theo dõi metrics
- **Phục vụ dự đoán**: API FastAPI cung cấp khả năng dự đoán real-time

## 🚀 Tính năng chính

- **Xử lý dữ liệu tự động**: Pipeline tiền xử lý và tạo đặc trưng từ dữ liệu thô
- **Huấn luyện mô hình**: XGBoost với tối ưu siêu tham số bằng Optuna
- **Model Registry**: Quản lý phiên bản mô hình với MLflow
- **API dự đoán**: FastAPI endpoint cho việc dự đoán real-time
- **Giao diện web**: Flask UI đơn giản để tương tác với hệ thống
- **Container hóa**: Docker Compose cho việc triển khai dễ dàng
- **Workflow orchestration**: Airflow DAG cho việc lập lịch và giám sát pipeline

## 🏗️ Kiến trúc hệ thống

```mermaid
graph TB
    A[Raw Data] --> B[Data Ingestion]
    B --> C[Data Processing]
    C --> D[Feature Engineering]
    D --> E[Model Training]
    E --> F[Model Registry]
    F --> G[API Service]
    G --> H[Web UI]

    I[Airflow] -.-> B
    I -.-> C
    I -.-> D
    I -.-> E

    J[MLflow] -.-> E
    J -.-> F

    K[PostgreSQL] -.-> I
    K -.-> J
```

### Thành phần chính:

- **MLflow**: Theo dõi thí nghiệm, lưu trữ mô hình và artifacts
- **Airflow**: Điều phối và chạy DAG `student_learning_pipeline`
- **FastAPI**: API backend phục vụ dự đoán với documentation tự động
- **Flask**: Giao diện web đơn giản cho người dùng cuối
- **PostgreSQL**: Cơ sở dữ liệu cho MLflow, Airflow và dữ liệu ứng dụng

## 💻 Yêu cầu hệ thống

- Docker Desktop hoặc Docker Engine + Docker Compose
- Python 3.10+ (nếu chạy local)
- 4GB RAM tối thiểu
- 10GB dung lượng ổ cứng

## 🛠️ Cài đặt và chạy

### Chạy với Docker Compose (Khuyến nghị)

1. **Khởi động hệ thống:**
   ```bash
   docker compose up --build -d
   ```

2. **Truy cập các service:**
   - MLflow UI: `http://localhost:5000`
   - Airflow UI: `http://localhost:8080`
   - FastAPI docs: `http://localhost:8000/docs`
   - Flask Web UI: `http://localhost:5001` (tránh xung đột với MLflow)

3. **Đăng nhập Airflow:**
   - Username: `admin`
   - Password: `admin123`

### Chạy local (Development)

1. **Cài đặt dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Khởi động các service riêng lẻ:**
   ```bash
   # Terminal 1: MLflow
   mlflow server --backend-store-uri postgresql://user:password@localhost/mlflow --default-artifact-root ./mlruns

   # Terminal 2: Airflow
   airflow webserver

   # Terminal 3: FastAPI
   uvicorn api.app:app --reload

   # Terminal 4: Flask
   python web_app.py
   ```

## 📖 Sử dụng

### Huấn luyện mô hình

**Cách 1: Qua Airflow UI**
1. Truy cập `http://localhost:8080`
2. Tìm DAG `student_learning_pipeline`
3. Click "Trigger DAG" để chạy pipeline

**Cách 2: Chạy trực tiếp**
```bash
docker compose exec airflow python /opt/airflow/project/main.py
```

### Dự đoán với API

```python
import requests

# Dự đoán hiệu suất học sinh
data = {
    "studied_credits": 120,
    "num_of_prev_attempts": 1,
    "gender": "M",
    "region": "Scotland",
    "highest_education": "A Level or equivalent",
    "imd_band": "20-30%",
    "age_band": "0-35",
    "disability": "N"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### Giao diện Web

Truy cập `http://localhost:5001` để sử dụng giao diện web đơn giản.

## 📁 Cấu trúc dự án

```
.
├── api/                    # FastAPI application
│   └── app.py
├── dags/                   # Airflow DAGs
│   └── ml_pipeline.py
├── data/                   # Dataset và processed data
│   ├── raw/               # Raw OULAD dataset
│   └── processed/         # Processed data và features
├── models/                # Trained models và metadata
├── src/                   # Source code
│   ├── ingestion/         # Data ingestion scripts
│   ├── pipeline/          # Pipeline orchestration
│   ├── processing/        # Data processing
│   ├── training/          # Model training
│   └── utils/             # Utility functions
├── templates/             # Flask HTML templates
├── static/                # CSS và JavaScript
├── tests/                 # Unit tests
├── postgres-init/         # Database initialization
├── Dockerfile.*           # Dockerfiles cho các service
├── docker-compose.yml     # Docker Compose configuration
├── requirements*.txt      # Python dependencies
└── README.md
```

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

### Hướng dẫn phát triển

- Sử dụng `black` để format code
- Viết tests cho các function mới
- Cập nhật documentation khi cần thiết
- Đảm bảo tất cả tests pass trước khi PR

## 📄 Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

---

**Lưu ý:** Bộ dữ liệu OULAD được cung cấp bởi Open University và chỉ sử dụng cho mục đích nghiên cứu và giáo dục.

```text
student_performance_model
```

### 4. Kiểm tra API

Sau khi đã có model trong MLflow, kiểm tra:

```bash
curl http://localhost:8000/health
```

Nếu API chưa nạp được model, gọi reload:

```bash
curl -X POST http://localhost:8000/reload-model
```

### 5. Sử dụng Flask Web UI

Truy cập `http://localhost:5000` để sử dụng giao diện web:

- **Input Fields**: Nhập dữ liệu học sinh (số click, ngày hoạt động, điểm trung bình, tín chỉ học)
- **Prediction**: Nhấn nút để dự đoán hiệu suất học tập
- **Visualization**: Xem biểu đồ phân phối hiệu suất mẫu
- **Health Check**: Kiểm tra trạng thái API trực tiếp trên trang

## API sử dụng

### `GET /`

Kiểm tra API đang chạy.

### `GET /health`

Trả về trạng thái model đã được nạp hay chưa.

### `POST /predict`

Payload:

```json
{
  "num_clicks": 420,
  "days_active": 18,
  "avg_score": 72.5,
  "studied_credits": 60
}
```

Ví dụ gọi API:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"num_clicks\":420,\"days_active\":18,\"avg_score\":72.5,\"studied_credits\":60}"
```

Ví dụ phản hồi:

```json
{
  "prediction": 1,
  "level": "Medium",
  "engagement_score": 0.5535,
  "consistency": 0.3
}
```

## Chạy local không dùng Docker

### 1. Tạo môi trường

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Cài dependencies

Pipeline core:

```bash
pip install -r requirements.txt
```

API:

```bash
pip install -r requirements.api.txt
```

### 3. Chạy MLflow local

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts
```

### 4. Chạy pipeline

```bash
set MLFLOW_TRACKING_URI=http://localhost:5000
python main.py
```

### 5. Chạy FastAPI

```bash
set MLFLOW_TRACKING_URI=http://localhost:5000
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Chạy Flask Web UI

```bash
pip install flask gunicorn
python web_app.py
```

Truy cập `http://localhost:5000` để sử dụng giao diện.

## Luồng pipeline

DAG `student_learning_pipeline` chạy 3 bước:

1. `preprocess`
2. `train_model`
3. `optuna_tuning`

Các file chính:

- [dags/ml_pipeline.py](dags/ml_pipeline.py)
- [src/pipeline/preprocess.py](src/pipeline/preprocess.py)
- [src/training/train.py](src/training/train.py)
- [src/training/optuna_tune.py](src/training/optuna_tune.py)

## Ghi chú kỹ thuật

- FastAPI và pipeline cùng dùng chung `MLFLOW_TRACKING_URI`.
- Feature engineering của API đã được đồng bộ với feature engineering khi train để tránh lệch đặc trưng giữa train và predict.
- Dữ liệu OULAD được gộp theo `code_module`, `code_presentation`, `id_student` để tránh trộn dữ liệu giữa các học phần.
- `Dockerfile.airflow` và `Dockerfile.mlflow` dùng file requirements riêng để tránh cài chồng dependency không cần thiết.

## Thành phần dữ liệu

Thư mục `data/raw/` hiện dùng các file:

- `studentInfo.csv`
- `studentVle.csv`
- `studentAssessment.csv`
- `assessments.csv`

Kết quả tiền xử lý được lưu tại:

```text
data/processed/train.csv
```

## Hạn chế hiện tại

- Thư mục `tests/` chưa có test tự động.
- API chỉ dự đoán sau khi model đã được train và được đưa lên stage `Production` trong MLflow.

## Tài liệu tham khảo

- OULAD Dataset: <https://analyse.kmi.open.ac.uk/open_dataset>
- FastAPI: <https://fastapi.tiangolo.com/>
- MLflow: <https://mlflow.org/docs/latest/index.html>
- Apache Airflow: <https://airflow.apache.org/docs/>
