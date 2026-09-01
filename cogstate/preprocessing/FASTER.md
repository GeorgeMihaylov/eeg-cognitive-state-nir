# MNE-FASTER: калибровка и потоковое применение

Проект использует `mne-faster==1.2.2` и `mne==1.10.2`. Собственная реализация
формул FASTER удалена.

`MNEFasterCalibrator` принимает эпохи в формате `[epochs, samples, channels]`,
создаёт `mne.EpochsArray` и вызывает публичные стадии `mne-faster`:

1. глобально плохие каналы;
2. плохие калибровочные эпохи;
3. ICA и статистически аномальные компоненты;
4. плохие каналы внутри оставшихся эпох и MNE-интерполяция;
5. average reference.

Результат включает очищенные калибровочные эпохи, маску сохранённых эпох,
диагностический отчёт и `MNEFasterBundle`. Bundle сохраняет MNE ICA, исключённые
компоненты, глобально плохие каналы, монтаж, единицы, порядок каналов и контракт
фильтрации.

```python
from cogstate.preprocessing import MNEFasterCalibrator, MNEFasterConfig

filter_contract = {
    "bandpass_low_hz": 1.0,
    "bandpass_high_hz": 45.0,
    "notch_hz": 50.0,
    "filter_mode": "causal",
}
config = MNEFasterConfig(
    sample_rate=256.0,
    channel_names=(
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
    ),
    input_scale_to_volts=1e-6,
    preprocessing_contract=filter_contract,
)
cleaned, bundle, report = MNEFasterCalibrator(config).fit_transform(epochs)
clean_labels = labels[report.kept_epoch_mask]
bundle.save("artifacts/calibration/user-001")
```

Полный FASTER остаётся пакетной калибровочной процедурой. Потоковый worker не
переобучает ICA и не пересчитывает глобальные решения. Он загружает bundle,
применяет фиксированную ICA, выявляет плохие каналы текущего окна средствами
`mne-faster`, интерполирует их через MNE и применяет average reference.

В `configs/streaming.yaml` очистка выключена по умолчанию:

```yaml
preprocessing:
  mne_faster_enabled: false
  mne_faster_bundle_dir: null
```

Для включения:

```yaml
preprocessing:
  mne_faster_enabled: true
  mne_faster_bundle_dir: artifacts/calibration/user-001
```

При включении bundle обязателен. Worker проверяет частоту дискретизации, порядок
каналов и контракт фильтрации. Модель также должна быть обучена с тем же режимом
очистки и содержать в манифесте `artifact_removal: mne_faster` и версию bundle.

Для 14-канальной гарнитуры пороги требуют отдельной абляции: референсный порог
`z=5` для глобальных каналов слишком консервативен при малом числе каналов.
Поэтому порог является явным параметром калибратора, а его значение должно быть
зафиксировано до внешней тестовой оценки.

Метод и реализация:

- Nolan, Whelan, Reilly, 2010, FASTER, DOI: 10.1016/j.jneumeth.2010.07.015.
- https://github.com/wmvanvliet/mne-faster
