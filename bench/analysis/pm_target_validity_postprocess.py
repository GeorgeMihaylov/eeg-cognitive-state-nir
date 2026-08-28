"""Postprocess PM target-validity outputs without model training."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from bench.analysis.pm_target_validity_cli import canonical_target_frame
from bench.analysis.pm_target_validity_audit import pm_column
from bench.tasks.target_registry import PM_METRICS
from bench.tasks.target_transforms import FoldLocalQuantileTargetTransform


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.')
    p.add_argument(
        '--audit-dir',
        default='reports/diagnostics/pm_target_validity_audit_full_v4',
    )
    p.add_argument(
        '--output-dir',
        default='reports/diagnostics/pm_target_validity_postprocess_v1',
    )
    a = p.parse_args()
    root = Path(a.root).resolve()
    out = root / a.output_dir
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(root / a.audit_dir / 'pm_raw_record_audit.csv')
    raw_rows = []
    for (source, metric), g in raw.groupby(['source', 'metric'], sort=True):
        phase = pd.to_numeric(
            g['phase_mean_seconds'], errors='coerce'
        ).dropna().to_numpy(float)
        z = np.mean(np.exp(1j * 2 * np.pi * phase / 10.0)) if len(phase) else np.nan
        b = pd.to_numeric(
            g['phase_boundary_distance_seconds'], errors='coerce'
        ).dropna()
        row = {
            'source': source,
            'metric': metric,
            'records': len(g),
            'valid_phase_records': len(phase),
            'record_phase_concentration': float(abs(z)) if len(phase) else np.nan,
            'phase_boundary_mean_s': float(b.mean()) if len(b) else np.nan,
            'implied_neighbor_mix_mean': float(b.mean() / 10) if len(b) else np.nan,
        }
        for c in [
            'raw_scaled_corr',
            'raw_to_scaled_slope',
            'raw_to_scaled_intercept',
            'raw_minmax_scaled_formula_mae',
            'raw_minmax_scaled_formula_match_fraction',
            'isactive_mean',
        ]:
            v = pd.to_numeric(g[c], errors='coerce').dropna()
            row[c + '_n'] = len(v)
            row[c + '_mean'] = float(v.mean()) if len(v) else np.nan
            row[c + '_median'] = float(v.median()) if len(v) else np.nan
            row[c + '_q25'] = float(v.quantile(.25)) if len(v) else np.nan
            row[c + '_q75'] = float(v.quantile(.75)) if len(v) else np.nan
        raw_rows.append(row)
    raw_summary = pd.DataFrame(raw_rows)
    raw_summary.to_csv(out / 'raw_robust_summary.csv', index=False)

    df, diag = canonical_target_frame(
        root / 'data/processed/windowed_eeg_pm_dataset_w10.parquet',
        root / 'data/interim/logical_recording_map.parquet',
        root / 'data/interim/raw_eeg_window_index_w10_raw_v3.parquet',
    )
    if len(df) != 30958:
        raise RuntimeError(f'canonical cohort mismatch: {len(df)}')

    overall = []
    qrows = []
    groups = df['subject_id'].astype(str).to_numpy()
    split = GroupKFold(5)
    for metric in PM_METRICS:
        mc = f'target_{metric}'
        lc = f"{pm_column(metric, 'Scaled')}__last"
        ac = f"{pm_column(metric, 'IsActive')}__mean"
        m = pd.to_numeric(df[mc], errors='coerce')
        l = pd.to_numeric(df[lc], errors='coerce')
        active = pd.to_numeric(df[ac], errors='coerce')
        pair = m.notna() & l.notna()
        d = (m[pair] - l[pair]).abs()
        iqr = m[pair].quantile(.75) - m[pair].quantile(.25)
        av = active[m.notna()].dropna()
        overall.append({
            'metric': metric,
            'mean_valid': int(m.notna().sum()),
            'last_valid': int(l.notna().sum()),
            'pairs': int(pair.sum()),
            'pearson': float(m[pair].corr(l[pair], method='pearson')),
            'spearman': float(m[pair].corr(l[pair], method='spearman')),
            'mean_last_mae': float(d.mean()),
            'mae_over_mean_iqr': float(d.mean() / iqr) if iqr > 0 else np.nan,
            'active_mean': float(av.mean()),
            'active_median': float(av.median()),
            'active_lt_0_5': float((av < .5).mean()),
            'active_lt_0_9': float((av < .9).mean()),
            'active_lt_0_99': float((av < .99).mean()),
            'active_lt_1': float((av < 1 - 1e-12).mean()),
        })

    for fold, (tr, te) in enumerate(
        split.split(np.zeros(len(df)), groups=groups), 1
    ):
        for metric in PM_METRICS:
            mc = f'target_{metric}'
            lc = f"{pm_column(metric, 'Scaled')}__last"
            mt = pd.to_numeric(
                df.iloc[tr][mc], errors='coerce'
            ).to_numpy(float)
            lt = pd.to_numeric(
                df.iloc[tr][lc], errors='coerce'
            ).to_numpy(float)
            me = pd.to_numeric(
                df.iloc[te][mc], errors='coerce'
            ).to_numpy(float)
            le = pd.to_numeric(
                df.iloc[te][lc], errors='coerce'
            ).to_numpy(float)
            a1 = FoldLocalQuantileTargetTransform(
                3, duplicates='drop'
            ).fit(mt)
            a2 = FoldLocalQuantileTargetTransform(
                3, duplicates='drop'
            ).fit(lt)
            y1 = a1.transform(me)
            y2 = a2.transform(le)
            ok = np.isfinite(y1) & np.isfinite(y2)
            delta = np.abs(y1[ok] - y2[ok])
            qrows.append({
                'fold': fold,
                'metric': metric,
                'pairs': int(ok.sum()),
                'q3_disagreement': float(np.mean(delta != 0)),
                'q3_severe_disagreement': float(np.mean(delta >= 2)),
            })

    overall = pd.DataFrame(overall)
    q = pd.DataFrame(qrows)
    qs = q.groupby('metric')[
        ['q3_disagreement', 'q3_severe_disagreement']
    ].agg(['mean', 'std']).reset_index()

    overall.to_csv(out / 'canonical_mean_last_isactive.csv', index=False)
    q.to_csv(out / 'q3_mean_last_by_fold.csv', index=False)
    qs.to_csv(out / 'q3_mean_last_summary.csv', index=False)

    print('cohort', diag)
    print('\nRAW ROBUST SUMMARY\n', raw_summary.to_string(index=False))
    print('\nMEAN/LAST + ISACTIVE\n', overall.to_string(index=False))
    print('\nQ3 MEAN/LAST\n', qs.to_string(index=False))


if __name__ == '__main__':
    main()
