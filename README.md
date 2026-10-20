# Bank Marketing Pareto HPO

Dự án nghiên cứu cá nhân về tối ưu đa mục tiêu cho bài toán dự đoán khách hàng đăng ký tiền gửi có kỳ hạn. Tuần 1–2 thiết lập protocol và leakage-safe baseline; Tuần 3 hoàn thành fresh AP–model-size HPO 40 Random + 40 TPE; Tuần 4 cung cấp notebooks, bảng và hình phân tích tái lập từ logs chính thức.

> **English short summary:** A reproducible Bank Marketing study with a leakage-safe baseline and an official train-only 40+40 multi-objective experiment. Random Search and TPE maximize mean 3-fold CV Average Precision while minimizing maximum-fold compressed pipeline size. Week 4 derives audited tables, Pareto analysis and publication-ready figures without retraining or reading the holdout.

## Problem

Dự đoán biến nhị phân `y` (`yes`/`no`) từ thông tin có thể dùng cho chiến dịch direct marketing. Do `duration` chỉ biết sau cuộc gọi và tạo leakage trong bối cảnh dự đoán trước cuộc gọi, cột này bị loại khỏi `X` **trước khi chia dữ liệu**.

`campaign`, `contact`, `day`, và `month` cũng cần được rà soát theo thời điểm ra quyết định thực tế: chúng có thể chỉ sẵn có khi chiến dịch/cuộc gọi đã được lên lịch hoặc đang diễn ra. Baseline giữ nguyên các cột này theo bộ dữ liệu công bố nhưng không khẳng định chúng luôn khả dụng khi triển khai. Xem [problem_formulation.md](docs/problem_formulation.md).

## Formulation

- Tuần 2: baseline cố định, tối đa hóa Average Precision (AP); báo thêm ROC-AUC, F1, precision và recall tại threshold `0.5`.
- Official AP–model-size v1: maximize mean 3-fold CV AP và minimize maximum serialized fitted-pipeline size qua ba folds.
- Ràng buộc: `max_fold_size_mb - 50.0 <= 0`, với decimal MB và Joblib compression 3.
- Latency không tham gia objective, feasibility, Pareto hoặc tie-breaking vì calibration v1/v2 không đạt stability gate.
- Holdout không được dùng cho HPO, Pareto, near-best selection hoặc Week 4 analysis.

Search space gốc nằm trong [experiment.yaml](configs/experiment.yaml); protocol chính thức, study isolation, budget và selection rule nằm trong [hpo_ap_model_size_v1.yaml](configs/hpo_ap_model_size_v1.yaml). Full `40+40` đã hoàn thành trong namespace `results/hpo/ap_model_size_v1/full_run_01/`.

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

### Official AP–model-size HPO

- Random Search: `optuna.samplers.RandomSampler(seed=42)`.
- TPE: Optuna 4.9.0 `TPESampler(seed=42, n_startup_trials=10, constraints_func=...)` với objectives native `maximize AP`, `minimize serialized size`.
- Random và TPE đều hoàn thành đúng 40 trials; 10 startup configurations trùng nhau do dùng cùng seed.
- Mỗi trial dùng đúng training partition và ba CV folds cố định. Holdout không được materialize thành `X_test/y_test` trong HPO.
- Mỗi fitted preprocessing-plus-estimator pipeline được serialize bằng Joblib compression 3; objective size là maximum decimal MB qua ba folds.
- SQLite cho pilot/full tách biệt và bị Git-ignore. Chạy lại cùng target sẽ resume mà không thêm trial.
- Chỉ các trial COMPLETE và feasible được đưa vào Pareto. Quy tắc chọn đã khóa giữ các nghiệm trong 0.005 AP của best rồi chọn size nhỏ nhất.

## Key results

| Reference | Mean CV AP ± std | Serialized size |
|---|---:|---:|
| Baseline | 0.4148607 ± 0.0074945 | 14.008766 MB full-train |
| Best Random, trial 31 | 0.4461627 ± 0.0043370 | 4.836080 MB CV max-fold |
| Best TPE, trial 38 | 0.4461862 ± 0.0045716 | 3.693059 MB CV max-fold |
| Selected TPE, trial 35 | 0.4434175 ± 0.0054617 | 1.034871 MB CV max-fold |
| Selected full-train audit | selection based on CV only | 1.289340 MB full-train |

All 80 trials completed and satisfied the 50 MB constraint. There are 70 distinct configurations, 13 distinct Pareto configurations, and 10 cross-sampler startup duplicates. The cap was non-binding: the largest observed CV max-fold size was 13.594303 MB.

The selected configuration is TPE trial 35, hash `4cdd6dd35f1d`. It gives up 0.0027687 absolute CV AP versus the best-AP point and reduces CV max-fold size by 71.978%. This is the registered trade-off decision, not the maximum-AP model.

With one sampler seed, the tiny Random–TPE best-AP difference is descriptive. It does not establish statistical superiority.

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

Pilot Tuần 3 (lệnh này resume; không nhân đôi nếu đã đủ 3 trials):

```powershell
.\.venv\Scripts\python.exe -m src.optimize --mode pilot --sampler both --trials 3
```

Official AP–model-size Full đã được thực hiện và không cần chạy lại để tái tạo
phân tích. Week 4 chỉ đọc logs hiện có:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_week4_figures.py
.\.venv\Scripts\python.exe .\scripts\build_week4_notebooks.py --execute
```

Outputs nằm trong `results/analysis/week4/`, `figures/week4/` và `notebooks/`.
Script xác minh CSV/JSON/SQLite, source hashes và figure hashes trước khi phân
tích. Không model training, HPO hoặc holdout evaluation nào được thực hiện.

Calibration latency train-only, version hóa (không chạy HPO):

```powershell
.\.venv\Scripts\python.exe -m src.latency_calibration --config configs\latency_calibration_v1.yaml
```

Đường AP–latency full bên dưới vẫn bị hard-block và chỉ được giữ để minh họa safety gate:

```powershell
.\.venv\Scripts\python.exe -m src.optimize --mode full --sampler both --trials 40 --confirm-full
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
- Week 3 AP–latency: pilot/calibration được giữ làm methodological diagnostics; full path vẫn bị block vì timing không ổn định.
- Week 3 AP–model-size v1: official fresh Full 40 Random + 40 TPE đã hoàn thành, 80/80 COMPLETE và feasible.
- Pareto chính thức: 13 cấu hình distinct; selected TPE trial 35, `4cdd6dd35f1d`.
- Latency calibration v1: đã chạy thật, 60 session summaries/900 raw repeats; gate **FAIL** vì baseline fold 3 có relative range 17,13% > 15%, dù không có contention flag hay measurement failure.
- Week 4: read-only analysis, 5 notebooks, 7 PNG 320 DPI + 7 PDF, source-integrity report và research limitations đã hoàn thành.
- Holdout không được đọc trong Week 3–4. Không có tuyên bố novelty hoặc TPE superiority.
- Week 5 chưa bắt đầu và cần nghiệm thu Week 4 trước.

## Roadmap

1. Tuần 3: hoàn thành AP–model-size Random Search 40 trials và Optuna TPE 40 trials trên cùng CV/search domain.
2. Tuần 4: hoàn thành kiểm tra integrity, phân tích tập khả thi/Pareto, notebooks và figures tái lập.
3. Tuần 5: sau nghiệm thu, tổng hợp báo cáo môn học; mọi multi-seed, multi-model hoặc external-test extension cần protocol riêng.

Xem [paper_roadmap.md](docs/paper_roadmap.md).

## Limitations

Một dataset và một sampler seed không chứng minh hiệu quả ngoài phân phối hay TPE superiority. CV fold standard deviation không thay thế phương sai giữa independent search runs. Holdout đã được xem ở Week 2 nên paper-quality final estimate cần external/temporal test mới. Joblib size phụ thuộc phiên bản và không đại diện cho latency, RAM hoặc energy. Cap 50 MB không binding. `unknown` là category được báo cáo, không tự động xem là missing. Chi tiết tại [week4_research_limitations.md](docs/week4_research_limitations.md).

## AI assistance

AI coding assistant hỗ trợ scaffold, code, tests và tài liệu. Tác giả chịu trách nhiệm kiểm tra dữ liệu, chạy lại thí nghiệm, review code, diễn giải kết quả và công bố minh bạch mức độ hỗ trợ theo quy định môn học/tạp chí. Không có số liệu fixture nào được dùng như kết quả nghiên cứu.
