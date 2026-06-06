# Results

Output results from model inference, analysis, and visualization.

## Organization

```
results/
├── predictions/
│   ├── segmentation_v1/
│   └── classification_v1/
├── visualizations/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── example_predictions/
├── metrics/
│   ├── model_v1_evaluation.csv
│   └── comparison_v1_vs_v2.json
└── reports/
    └── analysis_report_2026-05-05.md
```

## File Types

- `*.png, *.jpg` — Visualizations and example outputs
- `*.csv` — Metrics and evaluation results
- `*.json` — Structured data and configurations
- `*.md` — Analysis reports and documentation

## Naming Convention

Always include version/model name and date:
- `prediction_v1_segmentation_2026-05-05.png`
- `metrics_model_v2_2026-05-05.csv`

## Cleanup

Archive old results periodically. Keep recent results for comparison and reproducibility.
