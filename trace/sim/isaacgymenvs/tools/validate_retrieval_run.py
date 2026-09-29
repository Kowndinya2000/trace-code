"""Independently reconstruct expert, learned-stop, and baseline verdicts (CPU)."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(directory):
    directory=Path(directory)
    provenance=json.loads((directory/"provenance.json").read_text())
    complete=json.loads((directory/"complete.json").read_text())
    nominal=json.loads((directory/"nominal.json").read_text())
    assert complete["complete"]
    assert complete['protocol']==provenance['protocol']=='retrieval-expert-graspability-v4'
    rows=provenance["scene_rows"]
    n=len(rows)
    assert len(nominal['plans'])==len(nominal['rows'])==n
    assert len(set(r["sha256"] for r in rows))==n
    for row in rows: assert digest(row["path"])==row["sha256"]
    for p,v in zip(nominal["plans"],nominal["rows"]):
        assert len(p["xy"])==len(p["objects"])==len(p["actions"])+1
        if "graspability" in p:
            assert len(p["graspability"])==len(p["xy"])
            assert np.isfinite(p["graspability"]).all()
        assert len(p["actions"])==v["terminal_step"]
    reference=None
    summary={}
    for case in provenance["spec"]["cases"]:
        result=json.loads((directory/(case["name"]+".json")).read_text())
        assert result["complete"] and result["unique_scenes"]==n
        assert result['protocol']==provenance['protocol'] and len(result['rows'])==n
        stopping=result['termination']
        teacher_relative_budget=result.get('teacher_relative_budget')
        if teacher_relative_budget is not None:
            from isaacgymenvs.open_loop.teacher_relative_budget import validate as validate_budget
            teacher_relative_budget=validate_budget(teacher_relative_budget)
            assert case.get('teacher_relative_budget')==teacher_relative_budget
        assert stopping in ('nominal_action_budget','learned_score','expert_graspability',
                            'oracle_graspability','budget_rollout','budget_then_grasp')
        if stopping=='expert_graspability': assert case['actor']=='teacher'
        if stopping=='oracle_graspability':
            assert case['actor']=='student' and case.get('diagnostic_oracle')
            assert result['diagnostic_only'] and result['privileged_stopping']
        if stopping=='budget_rollout':
            assert case['actor']=='student' and case.get('diagnostic_oracle') and case.get('collect')
            assert result['diagnostic_only'] and not result['privileged_stopping']
        if stopping=='budget_then_grasp':
            assert case['actor']=='student' and case.get('diagnostic_oracle') and not case.get('collect')
            assert result['diagnostic_only'] and not result['privileged_stopping']
            assert float(case['budget_then_grasp_tolerance_m'])>=0
        if case.get('collect'):
            assert (case['actor'],stopping) in [('teacher','expert_graspability'),
                                               ('student','learned_score'),
                                               ('student','oracle_graspability'),
                                               ('student','budget_rollout')]
        assert result['success_scored_at']=='final_state_only' and result['simulated_retraction'] is False
        trace=directory/(case["name"]+".npz")
        assert digest(trace)==result["trace_sha256"]
        with np.load(trace) as archive:
            # Each array is decompressed once; validation arithmetic is unchanged.
            d={key:archive[key] for key in archive.files}
            live=d["live_before"].astype(bool)
            T=live.shape[0]
            assert live.shape[1]==n
            assert np.all(d["action"][~live]==-1)
            budgets=np.array([len(p['actions']) for p in nominal['plans']],np.int64)
            if case.get('fixed_plan_length') is not None:
                budgets=np.full(len(budgets),int(case['fixed_plan_length']),np.int64)
            np.testing.assert_array_equal(d['step_budget'],budgets)
            from isaacgymenvs.open_loop.obs_variants import obs_dim as _variant_obs_dim
            assert d['decision_obs'].shape==(T+1,n,_variant_obs_dim(result.get('obs_layout')))
            assert d['decision_valid'].shape==(T+1,n)
            np.testing.assert_allclose(d['decision_q'][0],d['initial_q'],equal_nan=True)
            np.testing.assert_allclose(d['decision_q'][1:],d['q'],equal_nan=True)
            np.testing.assert_array_equal(d['action_valid'][:-1],live)
            assert not d['action_valid'][-1].any()
            np.testing.assert_array_equal(d['decision_obs'][:-1],d['student_obs'])
            recovery_actor=result.get('cartesian_recovery_actor','student')
            assert recovery_actor in ('student','teacher')
            if recovery_actor=='teacher':
                assert case.get('cartesian_recovery_actor')=='teacher'
                assert case['actor']=='student' and case.get('diagnostic_oracle') and not case.get('collect')
                expected_teacher=d['action_valid'] & d['recovery_action']
                np.testing.assert_array_equal(
                    (d['cartesian_action_source']=='teacher_recovery') & d['action_valid'],
                    expected_teacher)
                np.testing.assert_array_equal(d['teacher_recovery_action'][:-1][expected_teacher[:-1]],
                                              d['action'][expected_teacher[:-1]])
                np.testing.assert_array_equal(d['cartesian_action_source'][d['action_valid'] & ~d['recovery_action']],
                                              'student_nominal')
            if reference is None:
                reference=(d["initial_state"].copy(),d["initial_eef"].copy(),d["initial_q"].copy())
            start_difference=float(np.max(np.abs(d["initial_state"]-reference[0])))
            pose_difference=float(np.max(np.abs(d["initial_state"][:,:,:7]-reference[0][:,:,:7])))
            q_difference=float(np.max(np.abs(d["initial_q"]-reference[2])))
            successes=0
            for i,row in enumerate(result["rows"]):
                budget=int(budgets[i])
                limit=(result['horizon'] if stopping=='budget_then_grasp' else
                       min(budget+teacher_relative_budget['extra_steps'],result['horizon'])
                       if teacher_relative_budget is not None else
                       budget if stopping=='nominal_action_budget' else result['horizon'])
                assert row['step_budget']==budget and row['terminal_step']<=limit
                assert not live[limit:,i].any(), 'Action executed beyond stop limit'
                np.testing.assert_array_equal(d['decision_valid'][:,i],np.arange(T+1)<=row['terminal_step'])
                reason=None
                lookups=failed_lookups=0
                remaining_queries=int(result.get('max_queries') or 0)
                last_lookup_q=last_lookup_step=last_lookup_travel=None
                extension_used=0.
                cartesian=result.get('cartesian_policy')
                for step in range(T+1):
                    if step==0:
                        q=float(d["initial_q"][i]);oow=bool(d["initial_oow"][i])
                        invalid=not (np.isfinite(d["initial_state"][i]).all() and np.isfinite(d["initial_eef"][i]).all())
                    else:
                        assert live[step-1,i], "Attempt retired before its recorded terminal"
                        q=float(d["q"][step-1,i]);oow=bool(d["mesh_oow"][step-1,i])
                        invalid=not (np.isfinite(d["block_state"][step-1,i]).all() and np.isfinite(d["eef"][step-1,i]).all())
                    if oow: reason="oow"
                    elif invalid or not np.isfinite(q): reason="invalid_state"
                    elif q==-2.: reason="out_of_view"
                    elif stopping=='budget_then_grasp':
                        assert teacher_relative_budget is not None
                        travel=float(d['tcp_travel_m'][step,i])
                        tolerance=float(case['budget_then_grasp_tolerance_m'])
                        length_limit=budget*teacher_relative_budget['primitive_length_m']+tolerance
                        if (step>=result['horizon'] or travel>=length_limit-1e-9
                                or travel+teacher_relative_budget['primitive_length_m']>length_limit+1e-9):
                            reason='success' if q>.9 else 'budget_ungraspable'
                        assert row['step_limit']==result['horizon']
                        np.testing.assert_allclose(row['length_limit_m'],length_limit,rtol=0,atol=1e-10)
                    elif stopping=='budget_rollout':
                        assert teacher_relative_budget is not None
                        travel=float(d['tcp_travel_m'][step,i])
                        step_limit=min(budget+teacher_relative_budget['extra_steps'],result['horizon'])
                        length_limit=budget*teacher_relative_budget['primitive_length_m']+teacher_relative_budget['extra_length_m']
                        if step>=step_limit: reason='step_timeout'
                        if travel>=length_limit-1e-9 or travel+teacher_relative_budget['primitive_length_m']>length_limit+1e-9:
                            reason='travel_timeout'
                        assert row['step_limit']==step_limit
                        np.testing.assert_allclose(row['length_limit_m'],length_limit,rtol=0,atol=1e-10)
                    elif stopping=='learned_score':
                        score=float(d['predicted_graspability'][step,i])
                        assert np.isfinite(score) and 0<=score<=1
                        if teacher_relative_budget is not None:
                            travel=float(d['tcp_travel_m'][step,i])
                            step_limit=min(budget+teacher_relative_budget['extra_steps'],result['horizon'])
                            length_limit=budget*teacher_relative_budget['primitive_length_m']+teacher_relative_budget['extra_length_m']
                            if step>=step_limit: reason='step_timeout'
                            if travel>=length_limit-1e-9 or travel+teacher_relative_budget['primitive_length_m']>length_limit+1e-9:
                                reason='travel_timeout'
                            assert row['step_limit']==step_limit
                            np.testing.assert_allclose(row['length_limit_m'],length_limit,rtol=0,atol=1e-10)
                        elif step==result['horizon']:
                            reason='timeout'
                        if result.get('lookup_feedback')=='numeric_q':
                            expected_context=np.zeros(6,np.float32)
                            expected_context[4]=min(step/max(step_limit,1),1.)
                            expected_context[5]=min(travel/max(length_limit,1e-6),1.)
                            if last_lookup_q is not None:
                                expected_context[:4]=[1.,last_lookup_q,
                                    min((step-last_lookup_step)/max(step_limit,1),1.),
                                    min((travel-last_lookup_travel)/max(length_limit,1e-6),1.)]
                            np.testing.assert_allclose(d['lookup_context'][step,i],expected_context,
                                                       rtol=0,atol=1e-6)
                            assert np.isfinite(d['predicted_current_q'][step,i])
                            assert np.isfinite(d['predicted_q_change'][step,i])
                            assert 0<=d['predicted_near_term_crossing'][step,i]<=1
                        decision_rule=result.get('stop_decision_rule')
                        if decision_rule in ('argmax_continue_stop','argmax_continue_query'):
                            values=d['stop_action_values'][step,i]
                            assert values.shape==(2,) and np.isfinite(values).all()
                            requested=bool(values[1]>values[0] and d['stop_eligible'][step,i])
                            if decision_rule=='argmax_continue_query':
                                assert result['max_queries']==3
                                assert int(d['query_budget_before'][step,i])==remaining_queries
                                assert bool(d['stop_eligible'][step,i])==(reason is None and remaining_queries>0)
                        else:
                            requested=bool(score>result['stop_threshold']
                                           and d.get('stop_eligible',d['decision_valid'])[step,i])
                        assert bool(d['stop_requested'][step,i])==requested
                        if requested and decision_rule=='argmax_continue_query':
                            lookups+=1
                            if q>.9 and reason is None:
                                reason='success'
                            elif q<=.9:
                                failed_lookups+=1
                                remaining_queries-=1
                                if result.get('lookup_feedback')=='numeric_q':
                                    last_lookup_q,last_lookup_step,last_lookup_travel=q,step,travel
                        elif requested:
                            reason='success' if q>.9 else 'premature_stop'
                    elif stopping in ('expert_graspability','oracle_graspability'):
                        if teacher_relative_budget is not None:
                            travel=float(d['tcp_travel_m'][step,i])
                            step_limit=min(budget+teacher_relative_budget['extra_steps'],result['horizon'])
                            length_limit=budget*teacher_relative_budget['primitive_length_m']+teacher_relative_budget['extra_length_m']
                            if step>=step_limit: reason='step_timeout'
                            if travel>=length_limit-1e-9 or travel+teacher_relative_budget['primitive_length_m']>length_limit+1e-9:
                                reason='travel_timeout'
                            if q>.9: reason='success'
                            assert row['step_limit']==step_limit
                            np.testing.assert_allclose(row['length_limit_m'],length_limit,rtol=0,atol=1e-10)
                        elif q>.9: reason='success'
                        elif step==result['horizon']: reason='horizon'
                    elif step==budget: reason='success' if q>.9 else 'budget_exhausted'
                    if result.get('plan_constraints') is not None:
                        from isaacgymenvs.open_loop.plan_constraint import select as constrained_select
                        eef=d['decision_obs'][step,i,110:112]
                        test_logits=np.zeros(16)
                        if step<T and live[step,i]: test_logits[int(d['action'][step,i])]=1.
                        extension=result.get('plan_extension')
                        chosen,details=constrained_select(test_logits,eef,nominal['plans'][i],step,result['plan_constraints'],extension,extension_used)
                        if extension is not None:
                            np.testing.assert_allclose(d['extension_travel_used'][step,i],extension_used,atol=1e-7)
                        if reason is None and chosen<0:
                            reason=details['reason']
                            assert str(d['constraint_reason'][step,i])==reason
                        elif reason is None:
                            assert live[step,i] and chosen==int(d['action'][step,i])
                            np.testing.assert_allclose(d['commanded_plan_offset'][step,i],details['commanded_max_offset_m'],atol=1e-7)
                            if extension is not None:
                                np.testing.assert_allclose(d['action_scale'][step,i],details.get('action_scale',1.),atol=1e-7)
                                extension_used+=details.get('extension_length_m',0.)
                                assert extension_used<=extension['max_length_m']+1e-7
                    if cartesian is not None and reason is None:
                        from isaacgymenvs.open_loop.teacher_relative_cartesian import decode as decode_cartesian
                        progress=int(d['plan_progress'][step,i])
                        used=float(d['measured_recovery_travel_used'][step,i])
                        eef=d['decision_obs'][step,i,110:112]
                        path,details=decode_cartesian(d['raw_cartesian_output'][step,i],eef,
                            nominal['plans'][i],progress,used,cartesian)
                        if path is None:
                            reason=details['reason']
                            assert str(d['constraint_reason'][step,i])==reason
                        else:
                            assert live[step,i]
                            np.testing.assert_allclose(d['executed_cartesian_path'][step,i],path,atol=1e-7)
                            assert bool(d['recovery_action'][step,i])==bool(details['recovery'])
                            assert used<=cartesian['max_recovery_travel_m']+1e-5
                            if not details['recovery']:
                                assert details['commanded_max_teacher_residual_m']<=cartesian['max_teacher_path_residual_m']+1e-7
                    if reason is not None:
                        if teacher_relative_budget is not None:
                            np.testing.assert_allclose(row['terminal_travel_m'],float(d['tcp_travel_m'][step,i]),
                                                       rtol=0,atol=1e-7)
                        assert row["reason"]==reason,(case["name"],i,row,reason)
                        assert row["terminal_step"]==step
                        np.testing.assert_allclose(row['terminal_q'],q,rtol=0,atol=1e-7,equal_nan=True)
                        assert not live[step:,i].any(), "Scored actions after terminal"
                        break
                assert reason is not None,"Unfinished scene"
                if result.get('stop_decision_rule')=='argmax_continue_query':
                    assert row['lookups']==lookups
                    assert row['failed_lookups']==failed_lookups
                    assert 0<=lookups<=result['max_queries']
                assert row["explicit_resets_during_rollout"]==0
                assert bool(row["success"])==(reason=="success")
                if 'ever_oracle_eligible_graspable' in row:
                    decision_bad = (d['decision_bad'][:,i] if 'decision_bad' in d
                                    else np.zeros_like(d['decision_valid'][:,i], dtype=bool))
                    expected_ever=bool(np.any((d['decision_q'][:,i]>.9)
                        &d['decision_valid'][:,i]&~decision_bad))
                    assert row['ever_oracle_eligible_graspable']==expected_ever
                    assert row['query_timing_failure']==bool(reason!='success' and expected_ever)
                if row['success'] and stopping=='nominal_action_budget': assert row['terminal_step']==budget
                successes+=reason=="success"
                # Visibility in each shuffled token must refer to its own object.
                order=np.r_[0,d["permutation"][i]+1]
                if T:
                    tokens=d["student_obs"][:,i,:110].reshape(T,11,10)
                    expected=d["visibility_world"][:,i,order]
                    np.testing.assert_array_equal(tokens[:,:,8],expected)
                    assert np.all(tokens[:,:,:8][expected==0]==0), "Hidden object geometry leaked"
                    if case.get("p_drop",0.)==1:
                        assert np.all(tokens[:,:,:9]==0)
            expected_pct=100*successes/n
            assert abs(expected_pct-result["success_pct"])<1e-8
            if case.get('collect'):
                with np.load(directory/'collection'/(case['name']+'.npz')) as labels:
                    assert str(labels['protocol'])==provenance['protocol']
                    assert str(labels['stopping'])==stopping and str(labels['actor'])==case['actor']
                    if stopping=='oracle_graspability': assert bool(labels['diagnostic_oracle'])
                    if 'visibility_config' in result:
                        assert json.loads(str(labels['visibility_config']))==result['visibility_config']
                    np.testing.assert_array_equal(labels['action_executed'],d['action_valid'])
                    if result.get('plan_constraints') is not None:
                        assert str(labels['bounded_collection_protocol'])=='bounded_collection_v1'
                        assert json.loads(str(labels['plan_constraints']))==result['plan_constraints']
                        assert json.loads(str(labels['plan_extension']))==result['plan_extension']
                        np.testing.assert_array_equal(labels['executed_action_scale'],d['action_scale'])
                        np.testing.assert_array_equal(labels['extension_travel_used'],d['extension_travel_used'])
                    if result.get('cartesian_policy') is not None:
                        assert str(labels['cartesian_collection_protocol'])=='teacher-relative-cartesian-collection-v1'
                        assert json.loads(str(labels['cartesian_policy']))==result['cartesian_policy']
                        for key in ('raw_cartesian_output','executed_cartesian_path','plan_progress',
                                    'recovery_action','measured_recovery_travel_used'):
                            np.testing.assert_array_equal(labels[key],d[key])
                    if teacher_relative_budget is not None:
                        assert str(labels['teacher_relative_budget_protocol'])==teacher_relative_budget['protocol']
                        assert json.loads(str(labels['teacher_relative_budget']))==teacher_relative_budget
                        np.testing.assert_array_equal(labels['tcp_travel_m'],d['tcp_travel_m'])
                    expected_stop=d['decision_valid']&np.isfinite(d['decision_q'])&np.isfinite(d['decision_obs']).all(-1)
                    np.testing.assert_array_equal(labels['stop_valid'],expected_stop)
                    expected_target=np.nan_to_num(np.where(d['decision_bad'],0,np.clip(d['decision_q'],0,1)))
                    np.testing.assert_array_equal(labels['graspability'],expected_target)
                    expected_action=d['action_valid']&np.isfinite(d['decision_q'])&(d['decision_q']<=.9)&(d['decision_q']!=-2.)&~d['decision_bad']
                    np.testing.assert_array_equal(labels['valid'],expected_action)
            summary[case["name"]]=dict(scenes=n,successes=successes,score=expected_pct,
                initial_state_max_abs_difference=start_difference,
                initial_pose_max_abs_difference=pose_difference,
                initial_q_max_abs_difference=q_difference)
    report=dict(valid=True,checks="scene/trace hashes, action-count budget, final-state-only success, budgeted queries/recovery/timeouts, no limit overrun, absorbing failures, visibility identity, missing geometry, exact counts",
                cases=summary,
                matched_initial_states=all(s["initial_state_max_abs_difference"]<=1e-5 and s["initial_q_max_abs_difference"]<=1e-5 for s in summary.values()))
    (directory/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("directory");args=ap.parse_args()
    print(json.dumps(validate(args.directory),indent=2))
