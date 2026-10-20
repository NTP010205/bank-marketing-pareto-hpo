# Bank Marketing Pareto HPO

Dự án nghiên cứu cá nhân về tối ưu đa mục tiêu cho bài toán dự đoán khách hàng đăng ký tiền gửi có kỳ hạn. Phạm vi hiện tại chỉ gồm **Tuần 1–2**: repository tái lập được, kiểm định dữ liệu và baseline Random Forest. Chưa chạy Random Search hay Optuna/TPE.

> **English short summary:** A reproducible Week 1–2 research scaffold for the UCI Bank Marketing binary-classification task. It audits the exact `bank-full.csv`, protects a stratified 20% test set, evaluates a leakage-safe Random Forest baseline with 3-fold CV, and records quality, latency, size, environment, and split provenance. Multi-objective HPO is deliberately deferred.

## Problem

Dự đoán biến nhị phân `y` (`yes`/`no`) từ thông tin có thể dùng cho chiến dịch direct marketing. Do `duration` chỉ biết sau cuộc gọi và tạo leakage trong bối cảnh dự đoán trước cuộc gọi, cột này bị loại khỏi `X` **trước khi chia dữ liệu**.

`campaign`, `contact`, `day`, và `month` cũng cần được rà soát theo thời điểm ra quyết định thực tế: chúng có thể chỉ sẵn có khi chiến dịch/cuộc gọi đã được lên lịch hoặc đang diễn ra. Baseline giữ nguyên các cột này theo bộ dữ liệu công bố nhưng không khẳng định chúng luôn khả dụng khi triển khai. Xem [problem_formulation.md](docs/problem_formulation.md).

## Formulation

- Tuần 2: baseline cố định, tối đa hóa Average Precision (AP); báo thêm ROC-AUC, F1, precision và recall tại threshold `0.5`.
- Tuần 3 (chỉ là protocol): minimize `(-AP_CV, latency_per_1000_samples)`.
- Ràng buộc đã đăng ký trước: model artifact không quá `50 MB`; latency không quá `1.5 ×` baseline đo thật.
- Nếu không có cấu hình khả thi, phải ghi log và giữ nguyên ràng buộc.

Search space và toàn bộ protocol được khóa trong [experiment.yaml](configs/experiment.yaml). Random Search và TPE dự kiến mỗi phương pháp 40 trials, cùng domain và folds, nhưng **chưa được triển khai/chạy**.

## Dataset attribution and license

Dữ liệu bắt buộc: **UCI Bank Marketing**, DOI [`10.24432/C5K306`](https://doi.org/10.24432/C5K306), giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Trang tải: <https://archive.ics.uci.edu/dataset/222/bank+marketing>.

Chỉ dùng `bank-full.csv` với delimiter `;`, shape kỳ vọng `(45211, 17)`. Archive có thể lồng theo chuỗi `bank+marketing.zip` → `bank.zip` → `bank-full.csv`. Raw data bị `.gitignore` loại khỏi Git. Chi tiết tại [DATASET.md](data/DATASET.md).

## Methods

- Holdout cố định: 20%, `train_test_split(..., stratify=y, random_state=42)`.
- Chỉ 80% train tham gia Stratified 3-fold CV.
- `ColumnTransformer` nằm trong `Pipeline`; imputation và one-hot encoding được fit lại trong từng training fold.
- `OneHotEncoder(handle_unknown='ignore')` cho categorical.
- `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)`.
- AP và ROC-AUC dùng `predict_proba`; threshold `0.5` chỉ dùng cho F1/precision/recall.
- Latency gồm preprocessing + `predict_proba`, 2 warm-ups, 7 repeats, batch 1000, giới hạn một thread; báo median/std/IQR.
- Kích thước model là kích thước file Joblib nén mức 3, không phải RAM.

## Experimental Protocol

Test set không được dùng cho EDA định hướng lựa chọn model, feature engineering, objective hay tuning. Baseline được định nghĩa trước trong config, đánh giá bằng CV trên train, sau đó mới fit trên toàn bộ train và đánh giá holdout một lần. Split indices, hashes, dataset SHA-256, timings, môi trường và phiên bản thư viện đều được lưu. Xem [experiment_protocol.md](docs/experiment_protocol.md).

## How to Run (Windows PowerShell)

Yêu cầu Python 3.11 và chạy từ thư mục repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Đặt file thật tại `data/raw/bank-full.csv`, sau đó:

```powershell
.\.venv\Scripts\python.exe -m src.data
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.benchmark
```

Hoặc chạy audit và baseline liên tiếp:

```powershell
.\scripts\run_week2.ps1
```

Các CLI không phụ thuộc current directory là `src`. Có thể override đường dẫn:

```powershell
.\.venv\Scripts\python.exe -m src.data --data D:\data\bank-full.csv --output results\audit.json
.\.venv\Scripts\python.exe -m src.benchmark --config configs\experiment.yaml --output results\baseline
```

## Current status

- Week 1 scaffold, attribution, protocol và dataset audit: implemented.
- Week 2 leakage-safe baseline, tests và result logging: implemented.
- Kết quả chạy thật nằm trong `results/data_audit.json` và `results/baseline/` sau khi chạy các lệnh trên; fixture tổng hợp chỉ dành cho tests và không phải kết quả UCI.
- Week 3–5: chỉ có config/stub/TODO, chưa có HPO hay tuyên bố Pareto.

## Roadmap

1. Tuần 3: chỉ sau khi baseline pass, triển khai Random Search 40 trials và Optuna TPE 40 trials trên cùng CV/search domain.
2. Tuần 4: áp ràng buộc size/latency đã khóa; phân tích tập khả thi và Pareto front.
3. Tuần 5: kiểm tra độ bền, tổng hợp bảng/hình và viết báo cáo/paper draft.

Xem [paper_roadmap.md](docs/paper_roadmap.md).

## Limitations

Một dataset lịch sử không chứng minh hiệu quả ngoài phân phối hay tính công bằng. `unknown` là category được báo cáo, không tự động xem là missing. Các biến liên quan lịch gọi cần kiểm tra availability theo hệ thống triển khai. Latency phụ thuộc phần cứng, phiên bản thư viện, nhiệt độ hệ thống và workload; số đo hiện tại chỉ là benchmark cục bộ có metadata.

## AI assistance

AI coding assistant hỗ trợ scaffold, code, tests và tài liệu. Tác giả chịu trách nhiệm kiểm tra dữ liệu, chạy lại thí nghiệm, review code, diễn giải kết quả và công bố minh bạch mức độ hỗ trợ theo quy định môn học/tạp chí. Không có số liệu fixture nào được dùng như kết quả nghiên cứu.
