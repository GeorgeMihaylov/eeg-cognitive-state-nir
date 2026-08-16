# CL-Drive: аудит и протокол мультимодального эксперимента

## Статус

Подготовлен воспроизводимый диагностический протокол без обучения моделей. Исходный ZIP и вложенные RAR прошли проверку SHA-256/CRC, распакованы с проверкой путей, исходный ZIP после операции не изменился.

## Архив и фактическая структура

- ZIP: `data/raw/cl_drive/archives/doi-10.5683-sp3-jj2yzz.zip`.
- Размер: 1 262 819 842 байта.
- SHA-256: `83862102235105dfb600e36706b36406094c6c7887e9d133a90042f54736aac0` — точное совпадение до и после распаковки.
- ZIP CRC и целостность пяти RAR подтверждены; небезопасных путей нет.
- В распакованном дереве 1 470 файлов общим объёмом 10 597 303 736 байт; файлов нулевого размера нет.
- Обнаружен один точный duplicate-hash pair: `Labels/1323.csv` и `Labels/1716.csv`. Файлы не удалялись и не объединялись; совпадение явно сохранено в `duplicate_file_audit.csv`.
- Фактически найдены 21 участник и 189 label-addressable task-записей. Полный primary EEG+ECG+EDA присутствует для 181 задачи; полный набор вместе с gaze — для 172.

## Контракты сигналов

EEG — числовой временной сигнал в CSV, четыре канала `TP9, AF7, AF8, TP10`, 256 Гц, медианный шаг 1/256 с. Длительности task-файлов 31,715–199,305 с. Вход ShallowConvNet — строго `[B,1,4,2560]` для 10-секундного окна. Единицы, референс и предшествующая acquisition/filtering history не подтверждены метаданными, поэтому эти свойства отмечены как неизвестные.

Peripheral contract:

- три калиброванных ECG-отведения, 512 Гц;
- основной калиброванный EDA conductance, 128 Гц.

Во многих файлах есть второй EDA sensor channel, но у участника `1716` его нет; он исключён ради стабильной одинаковой 28-признаковой схемы. Gaze — отдельный eye-tracking поток около 50 Гц и отсутствует для восьми task-записей, поэтому исключён из primary peripheral. Самостоятельные PPG/BVP, ACC и надёжный HR/IBI contract не обнаружены.

## Target, синхронизация и окна

Цель — субъективная оценка cognitive load 1–9 с явным столбцом времени и шагом 10 секунд. Для общего классификационного benchmark используется фиксированный `subjective_cognitive_load_3class_fixed`: 1–3 → 0, 4–6 → 1, 7–9 → 2. Пороговые группы заданы до folds и не вычисляются по данным; raw score сохраняется только как audit/target provenance и исключён из feature contract.

У CL-Drive начала EEG/ECG/EDA/gaze внутри задач близки и согласуются лучше, чем у CLARE. Тем не менее протокол консервативно не заявляет доказанный общий абсолютный clock и не применяет nearest-neighbour merge. Fusion выполняется по одному и тому же 10-секундному label interval в пределах парной participant/task записи, в относительном времени каждой модальности.

Общий cohort содержит 3 023 уникальных окна: low 802, medium 1 574, high 647. Материализация подтверждает `[3023,1,4,2560]` raw EEG, 52 EEG-признака и 28 peripheral-признаков; duplicate `sample_id` нет.

## Модели, признаки и folds

XGBoost имеет одинаковый фиксированный конфиг для CLARE и CL-Drive: 300 trees, depth 6, learning rate 0,05, subsample/colsample 0,8, `multi:softprob`, `hist`, seed 42. Режимы: EEG-only 52, peripheral-only 28, early fusion 80. Median imputer fit выполняется только на outer-train.

Shallow EEG-only использует существующий raw encoder без изменения архитектуры. Shallow fusion объединяет embedding EEG encoder (40) с отдельным peripheral MLP embedding (32), затем классифицирует 72-мерное представление. Все режимы оцениваются на идентичных sample IDs.

Используется 5-fold deterministic StratifiedGroupKFold по participant, seed 42. Во всех train/test частях есть все три класса, overlap участников равен нулю, каждое окно входит в test ровно один раз. Будущая матрица содержит 25 training/evaluation units: 15 XGBoost и 10 Shallow/fusion.

## Ограничения и рекомендация

CL-Drive — более сильный из двух кандидатов для первичного ShallowConvNet/fusion запуска: больше участников с полным EEG, явное время labels, больше завершённых задач и лучшее межмодальное согласование начала записи. Ограничения — всего 21 участник, субъективная 1–9 цель с инженерным 3-class mapping, недокументированная EEG acquisition history и исключение неполного второго EDA/gaze из primary contract. Результат остаётся внешней диагностической проверкой, а не самостоятельным окончательным доказательством универсальности мультимодального эффекта.

Protocol hash: `bb9aef380f8ed9edd19cfd8b565366ea0baf1d7855ab73dbe9f8e50e3de2bbbd`.

Команды:

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' scripts\run_external_multimodal_protocol.py --config experiments\external_datasets\cl_drive_multimodal_v1.json --plan-only
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' scripts\run_external_multimodal_protocol.py --config experiments\external_datasets\cl_drive_multimodal_v1.json --run-xgboost
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' scripts\run_external_multimodal_protocol.py --config experiments\external_datasets\cl_drive_multimodal_v1.json --run-shallow
```

Команды обучения только подготовлены; они не запускались.
