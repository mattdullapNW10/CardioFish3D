# Logs

Stores log files from training, processing, and evaluation.

## Organization

```
logs/
├── training_2026-05-05_15-30.log
├── preprocessing_2026-05-05_14-00.log
└── metrics/
    ├── training_metrics_v1.csv
    └── validation_metrics_v1.csv
```

## Log Levels

- DEBUG — Detailed information for diagnostics
- INFO — General informational messages
- WARNING — Warning messages for important events
- ERROR — Error messages

## Viewing Logs

```bash
# View recent logs
tail -n 100 logs/training_*.log

# Search logs
grep "ERROR" logs/training_*.log
```

## Cleanup

Old logs can be archived after analysis. Consider keeping logs for at least one month for reference.
