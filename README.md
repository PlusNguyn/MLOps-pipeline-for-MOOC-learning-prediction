# Dự Án Phân Tích Học Tập OULAD

## Tổng quan

Dự án này xây dựng một pipeline MLOps cho bộ dữ liệu OULAD (Open University Learning Analytics Dataset), gồm:

- tiền xử lý dữ liệu và tạo đặc trưng;
- huấn luyện mô hình XGBoost;
- tối ưu siêu tham số bằng Optuna;
- đăng ký mô hình vào MLflow Model Registry;
- phục vụ dự đoán qua FastAPI;
- điều phối pipeline bằng Airflow.

Phiên bản hiện tại đã được đồng bộ để `Airflow`, `MLflow` và `FastAPI` chạy cùng nhau ổn định trong Docker Compose.

## Thành phần hệ thống

- `MLflow`: theo dõi thí nghiệm, lưu model registry và artifact.
- `Airflow`: chạy DAG `student_learning_pipeline` để preprocess, train và tune model.
- `FastAPI`: nạp model từ MLflow Registry và cung cấp API dự đoán.
- `PostgreSQL`: backend store cho MLflow.

## Cấu trúc thư mục

```text
.
├── api/
│   └── app.py
├── dags/
│   └── ml_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── ingestion/
│   ├── pipeline/
│   ├── processing/
│   └── training/
├── Dockerfile.airflow
├── Dockerfile.api
├── Dockerfile.mlflow
├── docker-compose.yml
├── main.py
├── requirements.txt
├── requirements.api.txt
├── requirements.airflow.txt
└── requirements.mlflow.txt
```

## Yêu cầu

- Docker Desktop hoặc Docker Engine + Docker Compose
- Python 3.10+ nếu muốn chạy local

## Chạy nhanh bằng Docker Compose

### 1. Khởi động hệ thống

```bash
docker compose up --build -d
```

Các service sẽ chạy ở các địa chỉ:

- MLflow UI: `http://localhost:5000`
- Airflow UI: `http://localhost:8080`
- FastAPI docs: `http://localhost:8000/docs`

### 2. Đăng nhập Airflow

- Username: `admin`
- Password: `admin123`

### 3. Huấn luyện và đăng ký model

Có 2 cách:

1. Vào Airflow UI và trigger DAG `student_learning_pipeline`.
2. Hoặc chạy trực tiếp trong container Airflow:

```bash
docker compose exec airflow python /opt/airflow/project/main.py
```

Sau khi train xong, model sẽ được đăng ký vào MLflow Registry với tên:

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
