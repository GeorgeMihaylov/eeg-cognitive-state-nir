from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bench.experiments.pm_f3_f4_channel_drop import (
    VARIANTS, QC_VARIANTS, variant_profile, assert_same_index, retained_rows, aggregate, audit_predictions,
)
from bench.experiments.pm_low_high_q3_extremes_confirmatory import build_threshold_audit, PM_NAMES
from bench.features.cogstate_feature_cache import materialize_cogstate_features, load_feature_cache
from cogstate.features import FeaturePipeline, FeaturePipelineConfig
from cogstate.features import connectivity

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'experiments/features/preliminary_model_zoo_features_v1.json').read_text())


@pytest.mark.parametrize('variant,drop', [('drop_F3_13ch','F3'),('drop_F4_13ch','F4'),('drop_F8_13ch','F8'),('drop_F7_13ch','F7')])
def test_names_order_and_connectivity_recomputed(variant, drop, monkeypatch):
    profile, selected = variant_profile(PROFILE, variant)
    names = profile['feature_pipeline']['channel_names']
    assert names == [c for c in PROFILE['feature_pipeline']['channel_names'] if c != drop]
    assert len(names) == 13 and drop not in names
    pipeline = FeaturePipeline(FeaturePipelineConfig.from_mapping(profile))
    assert len(pipeline.feature_names()) == 346
    assert not any(n.endswith('__'+drop) for n in pipeline.feature_names())
    window = np.random.default_rng(4).normal(size=(14,2560)).astype(np.float32)
    window[PROFILE['feature_pipeline']['channel_names'].index(drop)] *= 100
    calls = []
    original = connectivity.extract_connectivity_features
    def spy(signal, config):
        calls.append(signal.shape[1])
        return original(signal, config)
    monkeypatch.setattr(connectivity, 'extract_connectivity_features', spy)
    output = pipeline.transform_window(window[selected].T)
    assert calls == [13] and np.isfinite(output).all()
    baseline = FeaturePipeline(FeaturePipelineConfig.from_mapping(PROFILE)).transform_window(window.T)
    assert not np.allclose(output[-21:], baseline[-21:])


def index_fixture():
    return pd.DataFrame([{'sample_id':s*3+i,'subject_id':f's{s}','record_id':f'r{s}',
                         'record_group_id':f'g{s}','outer_fold':s+1,'t_start':i*10.,'t_end':(i+1)*10.}
                        for s in range(5) for i in range(3)])


@pytest.mark.parametrize('column', ['sample_id','subject_id','outer_fold','t_start'])
def test_index_fold_pairing_mismatch_fails(column):
    frame = index_fixture()
    assert_same_index(frame, frame.copy())
    changed = frame.copy()
    changed.loc[1,column] = 'other' if column == 'subject_id' else 99
    with pytest.raises(ValueError, match='mismatch'):
        assert_same_index(frame, changed)


def test_fold_integer_storage_width_does_not_change_identity():
    frame = index_fixture()
    other = frame.copy()
    other['outer_fold'] = other.outer_fold.astype(np.int16)
    assert_same_index(frame, other)


def test_same_thresholds_and_retained_target_ids_across_variants():
    full = index_fixture()
    for pm in PM_NAMES:
        full['target_'+pm] = np.tile([.5, 0., 1.], 5)
    from bench.experiments.pm_eeg_lag_confirmatory import build_previous_window_pairing
    from bench.experiments.pm_low_high_q3_extremes_confirmatory import build_pm_temporal_cohorts
    pairing,_ = build_previous_window_pairing(full)
    cohorts,_ = build_pm_temporal_cohorts(full,pairing)
    transforms,audit = build_threshold_audit(full,cohorts,[1,2,3,4,5])
    context = SimpleNamespace(cohorts=cohorts,transforms=transforms)
    for variant in VARIANTS:
        for fold in range(1,6):
            for pm in PM_NAMES:
                for train in (True,False):
                    pd.testing.assert_frame_equal(retained_rows(context,fold,pm,train),retained_rows(deepcopy(context),fold,pm,train))
    assert len(audit)==35


def test_frozen_materialization_deterministic_and_no_cohort_rebuild(tmp_path, monkeypatch):
    from bench.features import cogstate_feature_cache as cache
    rng=np.random.default_rng(12)
    raw=rng.normal(size=(3,14,2560)).astype(np.float32)
    np.save(tmp_path/'raw.npy',raw)
    index=index_fixture().iloc[:3].copy()
    index['cache_file']=str(tmp_path/'raw.npy'); index['cache_offset']=np.arange(3)
    index['n_channels']=14; index['n_samples_expected']=2560
    index['status']='ok'; index['preprocessing_hash']='raw-frozen'
    monkeypatch.setattr(cache,'build_canonical_feature_index',lambda *a: pytest.fail('Must not rebuild cohort'))
    identities=[]
    for variant in (*VARIANTS[1:], *QC_VARIANTS[1:]):
        profile,selected=variant_profile(PROFILE,variant)
        profile_path=tmp_path/(variant+'.json'); profile_path.write_text(json.dumps(profile))
        kwargs=dict(manifest_path=tmp_path/'unused',logical_recording_map_path=tmp_path/'unused',
                    cache_path_root=tmp_path,feature_profile_path=profile_path,frozen_index=index,
                    raw_channel_indices=selected,source_cache_identity='baseline-id',chunk_size=2)
        matrices=[]
        for suffix in ('a','b'):
            output=tmp_path/(variant+suffix)
            if suffix == 'a':
                original_batch = cache._window_batch
                def interrupted_batch(view, start, stop):
                    if start >= 2:
                        raise RuntimeError('simulated interruption after first saved block')
                    return original_batch(view, start, stop)
                monkeypatch.setattr(cache, '_window_batch', interrupted_batch)
                with pytest.raises(RuntimeError, match='simulated interruption'):
                    cache.materialize_cogstate_features(**kwargs,output_dir=output)
                monkeypatch.setattr(cache, '_window_batch', original_batch)
                progress=json.loads((output/'feature_materialization_manifest.json').read_text())
                assert progress['completed_rows']==2
                cache.materialize_cogstate_features(**kwargs,output_dir=output,resume=True)
            else:
                cache.materialize_cogstate_features(**kwargs,output_dir=output)
            matrix,stored,names,manifest=load_feature_cache(output)
            assert_same_index(index,stored)
            assert matrix.shape==(3,346) and matrix.dtype==np.float32
            assert manifest['identity']['input_transform']['indices']==selected
            before=matrix.copy()
            cache.materialize_cogstate_features(**kwargs,output_dir=output,resume=True)
            np.testing.assert_array_equal(before,load_feature_cache(output)[0])
            matrices.append(matrix.copy())
            identities.append(manifest['identity']['cache_identity_hash'])
        np.testing.assert_array_equal(*matrices)
    assert identities[0]==identities[1] and identities[2]==identities[3] and identities[0]!=identities[2]


def test_participant_bootstrap_not_folds_and_cohort_mismatch():
    rows=[]
    for variant,delta in zip(VARIANTS,[0,.02,-.03]):
        for pm in PM_NAMES:
            for s in range(6):
                rows.append(dict(variant=variant,pm=pm,subject_id=str(s),outer_fold=s%5+1,n_samples=10,
                                 balanced_accuracy=.6+delta,macro_f1=.5+delta,roc_auc=.7+delta,pr_auc=.65+delta,
                                 low_recall=.6+delta,high_recall=.6+delta))
    frame=pd.DataFrame(rows)
    cfg={'targets':list(PM_NAMES),'bootstrap':{'unit':'subject','n_resamples':100,'random_state':42,'confidence_level':.95}}
    summary,deltas,ci=aggregate(frame,cfg)
    assert len(summary)==24 and ci.n_subjects.eq(6).all()
    np.testing.assert_allclose(ci.loc[ci.variant.eq('drop_F3_13ch'),'mean_delta_ba'], .02)
    np.testing.assert_allclose(ci.loc[ci.variant.eq('drop_F4_13ch'),'ci_high'],-.03)
    assert ci.loc[ci.variant.eq('drop_F4_13ch'),'fraction_participants_worsened'].eq(1).all()
    assert ci.fraction_participants_tied.eq(0).all()
    np.testing.assert_allclose(deltas.drop_balanced_accuracy-deltas.baseline_balanced_accuracy,
                               deltas.delta_balanced_accuracy)
    with pytest.raises(ValueError,match='cohort mismatch'):
        aggregate(frame.drop(index=len(frame)-1),cfg)


@pytest.mark.parametrize('mutation', ['feature_id', 'target_id', 'threshold', 'subject'])
def test_saved_baseline_prediction_mismatch_rejected(tmp_path, mutation):
    from bench.experiments.pm_low_high_q3_extremes_confirmatory import ExtremeThresholds
    cohort = pd.DataFrame({'target_sample_id':[2,3], 'lag_minus_10s_feature_sample_id':[0,1],
                          'subject_id':['s','s'], 'record_id':['r','r'], 'outer_fold':[1,1],
                          'continuous_target':[.1,.9]})
    context=SimpleNamespace(cohorts={'focus':cohort},transforms={(1,'focus'):ExtremeThresholds(.3,.7)})
    saved=pd.DataFrame({'target_sample_id':[2,3], 'feature_sample_id':[0,1], 'subject_id':['s','s'],
                        'record_id':['r','r'], 'outer_fold':[1,1], 'y_true':[0,1], 'y_pred':[0,1],
                        'probability_high':[.1,.9], 'q_low':[.3,.3], 'q_high':[.7,.7]})
    path=tmp_path/'predictions.parquet'
    spec={'outer_fold':1,'pm':'focus','q_low':.3,'q_high':.7}
    saved.to_parquet(path,index=False)
    participant,_=audit_predictions(context,spec,path)
    assert participant.balanced_accuracy.item()==1
    column={'feature_id':'feature_sample_id','target_id':'target_sample_id','threshold':'q_low','subject':'subject_id'}[mutation]
    saved.loc[0,column]='wrong' if mutation=='subject' else 99
    saved.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='mismatch'):
        audit_predictions(context,spec,path)


def test_existing_execution_path_accepts_only_explicit_346_contract(tmp_path):
    from test_pm_low_high_q3_extremes_confirmatory import _synthetic_context
    from bench.experiments.pm_low_high_q3_extremes_confirmatory import execute_run
    context=_synthetic_context(tmp_path)
    context.matrix=context.matrix[:,:346]
    context.feature_names=context.feature_names[:346]
    context.cache_identity=dict(context.cache_identity,n_features=346)
    spec=context.run_matrix.iloc[0].to_dict()
    class Dummy:
        classes_=np.array([0,1])
        def fit(self,x,y):
            assert x.shape[1]==346
        def predict(self,x):
            return np.zeros(len(x),dtype=int)
        def predict_proba(self,x):
            return np.tile([.6,.4],(len(x),1))
    def builder(*args):
        assert args[2]==(346,)
        return Dummy()
    with pytest.raises(RuntimeError,match='371'):
        execute_run(context,spec,model_builder=builder)
    summary=execute_run(context,spec,model_builder=builder,expected_feature_count=346,result_status='diagnostic')
    assert summary['result_status']=='diagnostic'


@pytest.mark.parametrize('mutation', ['fold', 'threshold', 'pairing'])
@pytest.mark.parametrize('variants', [VARIANTS, QC_VARIANTS])
def test_pretraining_contract_all_masks_and_fail_fast(tmp_path, mutation, variants):
    from test_pm_low_high_q3_extremes_confirmatory import _synthetic_context
    from bench.experiments.pm_f3_f4_channel_drop import pretraining_contract
    context = _synthetic_context(tmp_path)
    context.feature_index['t_end'] = context.feature_index.t_start + 10
    representations = {}
    for variant in variants:
        profile, _ = variant_profile(PROFILE, variant)
        names = FeaturePipeline(FeaturePipelineConfig.from_mapping(profile)).feature_names()
        representations[variant] = (np.zeros((len(context.feature_index), len(names))),
            context.feature_index.copy(), names,
            {'identity': {'channel_order': profile['feature_pipeline']['channel_names'], 'cache_identity_hash': variant}})
    audit = pretraining_contract(context, representations)
    assert len(audit) == 105
    assert {'train_low_target_hash', 'test_high_target_hash', 'test_middle_target_hash'} <= set(audit.columns)
    if mutation == 'fold':
        representations[variants[1]][1].loc[0, 'outer_fold'] = 99
    elif mutation == 'threshold':
        context.run_matrix.loc[0, 'q_low'] += .1
    else:
        context.cohorts['focus'].loc[0, 'lag_minus_10s_feature_sample_id'] = -1
    with pytest.raises(ValueError, match='mismatch'):
        pretraining_contract(context, representations)
