# Структура проекта

```text
eeg-cognitive-state-nir/
├── bench/
│   ├── core/
│   │   ├── abstract_dataset.py
│   │   └── abstract_task.py
│   ├── datasets/
│   │   ├── base_eeg_data_loader.py
│   │   ├── emotiv_loader.py
│   │   └── datasets_registry.py
│   ├── tasks/
│   │   ├── cognitive_load.py
│   │   └── tasks_registry.py
│   ├── preprocessing/
│   │   ├── filters.py
│   │   ├── artifacts.py
│   │   ├── features.py
│   │   └── preprocessing_pipeline.py
│   ├── validation/
│   │   ├── metrics.py
│   │   └── cross_val.py
│   └── bench_runner.py
├── configs.yaml
├── cli.py
└── README.md
```

# Запуск

## Управление через CLI
| Аргумент | Описание |
|----------|----------|
| `--config` | Путь к YAML файлу конфигурации |
| `--dataset` | Запустить только указанный датасет |
| `--models` | Запустить только указанные модели (через запятую) |
| `--task` | Запустить только указанную задачу |
| `--output-dir` | Директория для сохранения результатов |
| `--feature-set` | Набор признаков (`pow_plus_eeg`, `pow`, `eeg`, `all`) |
| `--no-loso` | Отключить LOSO-валидацию |
| `--no-within` | Отключить within-subject валидацию |

## Примеры

```
# Запуск с заданной конфигурацией
python cli.py --config configs.yaml

# Запуск конкретной модели (Random Forest)
python cli.py --config configs.yaml --models random_forest

# Запуск только с LOSO валидацией
python cli.py --config configs.yaml --no-within
```