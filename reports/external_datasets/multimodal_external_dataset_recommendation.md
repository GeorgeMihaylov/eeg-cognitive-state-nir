# Общая рекомендация по внешней мультимодальной проверке

CLARE и CL-Drive технически пригодны для participant-disjoint сравнения EEG-only, peripheral-only и EEG+peripheral на одном cohort внутри каждого датасета. Они не смешиваются между собой: у каждого отдельные manifest, folds, protocol hash, run matrix и будущие результаты.

Полные XGBoost и ShallowConvNet/fusion эксперименты выполнены на CL-Drive и
CLARE. CL-Drive остаётся методически более сильным из двух наборов: 21
участник против 19 пригодных в CLARE, явные 10-секундные label timestamps, 181
полная primary task-запись и более согласованные межмодальные начала. CLARE
полезен как второй независимый diagnostic, но его label times выводятся из
порядка строк, а clocks модальностей явно различаются.

Совместный результат не поддерживает универсальное улучшение fusion: XGBoost
даёт ΔMacro F1 +0.113961 на MEFAR и +0.011120 на CL-Drive, но −0.037978 на
CLARE; Shallow fusion даёт −0.070163 на CL-Drive и −0.112538 на CLARE. На
MEFAR peripheral-only (0.577597) также превосходит fusion (0.511133). Поэтому
эффект зависит от dataset и модели. MEFAR сохраняет target
`mefar_cfs_fatigue_binary`; raw EEG там отсутствует, поэтому ShallowConvNet не
применяется.

Немедленный поиск ещё одного датасета до выполнения этих двух планов не требуется. Если после них нужен более сильный внешний вывод, следующий набор следует выбирать с другой acquisition setup и объективно подтверждёнными timestamps/labels; WESAD без EEG не является эквивалентной проверкой, а Digit Span или SEED-VIG требуют отдельного согласования target semantics.

Итоговая интерпретация должна сохранять отдельно dataset/model/modality,
Macro F1 и Balanced Accuracy по folds, а также парные `ΔMacroF1` fusion −
EEG-only. Усреднять эти гетерогенные задачи в один «универсальный» эффект без
отдельной статистической модели нельзя.
