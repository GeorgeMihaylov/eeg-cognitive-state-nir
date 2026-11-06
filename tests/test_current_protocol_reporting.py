from types import SimpleNamespace
import yaml
import pytest
from bench.analysis import experiment_summary as summary
from bench.analysis import colleague_metrics_package as colleague
from bench.analysis import experiment_config_audit as config
from test_experiment_summary import entry, registry, prepare_root


def test_current_and_legacy_rankings_are_separate(tmp_path):
    prepare_root(tmp_path)
    old=entry('old',status='final'); old['scientific_status']='legacy'
    current=entry('current',status='final'); current.update(scientific_status='current_primary',task='seven_pm_low_high',target='all_seven_pm')
    built=summary.build_summaries(registry(old,current),tmp_path,strict=True)
    report=summary.render_experiment_markdown(built)
    main, legacy=report.split('## Historical / legacy',1)
    assert 'Эксперимент old' not in main
    assert 'Эксперимент current' in main
    assert 'Эксперимент old' in legacy
    rows={x['experiment_id']:x for x in built['experiment_rows']}
    assert rows['old']['primary_value']==rows['current']['primary_value']==.5
    current['scientific_status']='pretend_current'
    with pytest.raises(summary.RegistryError,match='scientific_status'):
        summary.build_summaries(registry(current),tmp_path)


def test_colleague_current_preface_preserves_historical_package(monkeypatch):
    monkeypatch.setattr(colleague,'_report',lambda *args:'old exact numerical table: 0.3570\n')
    report=colleague.render_current_and_historical_report({'current_summary_markdown':'seven-PM LOW/HIGH current'},None,None,None)
    assert report.index('seven-PM LOW/HIGH')<report.index('Historical package')<report.index('0.3570')
    assert report.endswith('old exact numerical table: 0.3570\n')


def test_dedicated_json_protocol_metadata_is_not_claimed_as_audited():
    item={'config_path':'experiment.json','scientific_status':'current_primary','metadata_only':True}
    result=SimpleNamespace(records=[],duplicate_groups=[],curation={'reviewed_protocol_metadata':[item]})
    payload=yaml.safe_load(config.render_config_registry(result))
    assert payload['configs']==[item]
    assert 'resolved_config_hash' not in payload['configs'][0]


def test_historical_final_renderer_cannot_overwrite_current_reports():
    from pathlib import Path
    from bench.analysis import project_final_package as package
    root=Path(__file__).resolve().parents[1]
    inventory=package.build_inventory(root)
    rendered=package._render_reports(root,inventory,package.build_provenance_audit(root,inventory),
        package.build_requirement_rows(root),package.build_negative_results(),
        package.build_meta_learning_results(root),package.build_domain_adaptation_results(root),
        package.build_experiment_statuses(root),package.build_lag_alignment_summary(root))
    assert root/'reports/integration/project_scientific_conclusions.md' not in rendered
    assert root/'reports/requirements/final_requirement_coverage.md' not in rendered
    assert rendered[root/'reports/integration/project_final_state.md'].startswith('> **HISTORICAL SNAPSHOT')


def test_text_audit_keeps_tertile_boundaries_distinct_from_q3():
    from scripts.analysis.audit_current_protocol_documentation import scan_text, classify
    row=scan_text('example.py','Q33 Q67\npreprocessing_q3_v1 label_q5\nthree-class\n')
    assert row['matches']=={'Q3':1,'label_q5':1,'3-class':1}
    assert classify('reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1/README.md',before=False,text='Q3 namespace')[0]=='current_correct'
    assert classify('experiments/cog_bci/native.yaml',before=False,text='nback_3class')[0]=='native_three_class_keep'
