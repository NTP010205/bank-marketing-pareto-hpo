# Dataset card — UCI Bank Marketing

## Identity and attribution

- Dataset: UCI Bank Marketing
- Download page: <https://archive.ics.uci.edu/dataset/222/bank+marketing>
- DOI: <https://doi.org/10.24432/C5K306>
- License: CC BY 4.0
- Required file: `bank-full.csv`
- Expected shape: 45,211 rows × 17 columns
- Delimiter: semicolon (`;`)
- Target: `y`, exactly `yes`/`no`

The raw dataset is not redistributed by this repository. Cite the UCI record and its named creators in any report or paper, using the citation supplied by the repository landing page.

## Installation

Download `bank+marketing.zip`. The archive can contain another archive; extract:

```text
bank+marketing.zip
└── bank.zip
    └── bank-full.csv
```

Place that exact file at `data/raw/bank-full.csv`. Do not substitute the smaller `bank.csv` or the differently defined `bank-additional-full.csv`. `.gitignore` prevents the raw CSV from being committed.

## Automated audit

```powershell
.\.venv\Scripts\python.exe -m src.data
```

The audit checks schema, exact shape, labels, nulls, literal `unknown` values by column, class counts/proportions, duplicate rows, byte size, and SHA-256. `unknown` is retained as a disclosed categorical value; it is not silently converted to a null.

## Leakage and availability

`duration` is known only after a call and is always excluded before train/test splitting. `campaign`, `contact`, `day`, and `month` may have different availability depending on whether prediction occurs during planning, immediately before a call, or during a campaign. The current baseline retains them but records this deployment-risk assumption. No extra features are inferred or synthesized for the real experiment.
