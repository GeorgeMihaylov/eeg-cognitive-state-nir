import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.analysis import channel_quality as qc

CONFIG = json.loads((Path(__file__).resolve().parents[1] / qc.SPEC).read_text())


def test_fixed_channel_order_and_determinism():
    assert qc.CHANNELS == ('AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4')
    x = np.random.default_rng(42).normal(size=(14,2560)).astype(np.float32)
    pd.testing.assert_frame_equal(qc.window_quality(x, CONFIG, {}), qc.window_quality(x, CONFIG, {}))
    assert qc.window_quality(x, CONFIG, {}).channel.tolist() == list(qc.CHANNELS)


def test_constant_signal_and_zero_denominators():
    result = qc.window_quality(np.zeros((14,2560)), CONFIG, {})
    assert result['std'].eq(0).all() and result.flat_fraction.eq(1).all()
    assert result.exact_repeat_fraction.eq(1).all()
    assert result.roughness_ratio.isna().all() and result.rel_delta.isna().all()
    assert result.z_std.isna().all() and result.amplitude_violation.isna().all()


def test_noise_amplitude_and_existing_thresholds():
    rng = np.random.default_rng(19)
    t = np.arange(2560)/256
    x = np.stack([(1+i*.1)*np.sin(2*np.pi*10*t)+rng.normal(0,.01,2560) for i in range(14)])
    x[0] *= 100
    x[1] = rng.normal(0,1,2560)
    result = qc.window_quality(x, CONFIG, {'max_abs_amplitude':10,'max_flat_fraction':.9})
    assert result.loc[0,'anomaly_std_high']==1 and result.loc[0,'amplitude_violation']==1
    assert result.loc[1,'roughness_ratio'] > 1 and result.loc[1,'anomaly_roughness_ratio_high']==1
    assert result.loc[1,'high_frequency_ratio'] > result.loc[2,'high_frequency_ratio']


def test_robust_leave_one_out_score():
    values = np.arange(14,dtype=float)
    peers = values[1:]
    expected = (values[0]-np.median(peers))/np.median(abs(peers-np.median(peers)))
    assert qc.robust_peer_scores(values)[0] == expected
    assert np.isnan(qc.robust_peer_scores(np.ones(14))).all()


def test_nonfinite_channel_not_imputed():
    x = np.random.default_rng(4).normal(size=(14,2560))
    x[0,1]=np.nan; x[1,2]=np.inf
    result = qc.window_quality(x,CONFIG,{})
    assert result.loc[0,'nonfinite_fraction']==pytest.approx(1/2560)
    assert result.loc[:1,'std'].isna().all()
    assert result.loc[2:,'std'].notna().all()


@pytest.mark.parametrize('shape',[(13,2560),(2560,14),(14,2559)])
def test_shape_mismatch(shape):
    with pytest.raises(ValueError,match='shape mismatch'):
        qc.window_quality(np.zeros(shape),CONFIG,{})


def test_record_then_participant_aggregation_equal_record_weight():
    rows=[]
    for subject, record, n, value in [('a','a1',100,0),('a','a2',1,10),('b','b1',2,20)]:
        for i in range(n):
            for channel in qc.CHANNELS:
                rows.append(dict(subject_id=subject,record_id=record,record_group_id=record,outer_fold=1,
                                 sample_id=f'{record}_{i}',channel=channel,std=value,
                                 anomaly_std_high=float(value>5)))
    record,subject,glob,fold=qc.summarize(pd.DataFrame(rows))
    assert len(record)==42 and len(subject)==28 and len(glob)==14
    assert subject.loc[subject.subject_id.eq('a'),'std'].eq(5).all()
    assert glob.participant_median_std.eq(12.5).all()
    assert glob.participant_median_anomaly_std_high.eq(.75).all()


def test_frozen_universe_order_and_mismatch():
    index=pd.DataFrame([dict(sample_id=i,subject_id='s',record_id='r',record_group_id='r',outer_fold=1,t_start=i*10,t_end=(i+1)*10) for i in range(3)])
    raw=index.iloc[::-1].copy()
    pd.testing.assert_frame_equal(qc.align_frozen(index,raw),index)
    raw.loc[1,'outer_fold']=2
    with pytest.raises(ValueError,match='metadata mismatch'):
        qc.align_frozen(index,raw)


def test_no_label_or_training_dependency():
    source=inspect.getsource(qc.run)
    assert 'processed_targets' not in source and 'run_preflight(' not in source
    assert 'validate_frozen_raw_inputs(' in source
    assert '.fit(' not in source and 'build_model' not in source
    x=np.random.default_rng(2).normal(size=(14,2560))
    config=dict(CONFIG, target_focus=[999], label_q5=[-1])
    pd.testing.assert_frame_equal(qc.window_quality(x,config,{}),qc.window_quality(x,CONFIG,{}))
