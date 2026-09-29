"""Regression checks for scoring, scene identity, and missing observations."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import ast
import math
import numpy as np


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+".py"))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


core,so=load("evaluation_core"),load("student_obs")


class EvaluationContracts(unittest.TestCase):
    def test_decoder_vectors_match_actual_executor(self):
        import torch
        p=Path(__file__).resolve().parents[1]/"tasks/more.py"
        tree=ast.parse(p.read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="build_two_step_plan")
        namespace=dict(torch=torch,math=math)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),"exec"),namespace)
        d1,d2=namespace["build_two_step_plan"](None,torch.arange(16),total=.04)
        np.testing.assert_allclose((d1+d2).numpy(),core.primitive_vectors(.04),atol=1e-8)

    def test_failure_cannot_be_repaired_by_a_retry(self):
        l=core.FirstAttempt(2,3)
        l.update(1,[-2.,.2],[False,False])
        l.update(2,[.99,.99],[False,False])
        self.assertEqual(l.reason.tolist(),["out_of_view","success"])
        self.assertEqual(l.step.tolist(),[1,2])

    def test_later_violation_cannot_revoke_success(self):
        l=core.FirstAttempt(2,3)
        l.update(1,[.99,.2],[False,False])
        l.update(2,[.99,.99],[True,False])
        self.assertEqual(l.reason.tolist(),["success","success"])

    def test_simultaneous_violation_wins(self):
        l=core.FirstAttempt(1,3)
        l.update(1,[.99],[True])
        self.assertEqual(l.reason[0],"oow")

    def test_horizon_and_replay_exhaustion(self):
        l=core.FirstAttempt(3,2)
        l.update(1,[.1,.1,.1],[False]*3,exhausted=[False,True,False])
        l.update(2,[.1,.99,float('nan')],[False]*3)
        self.assertEqual(l.reason.tolist(),["horizon","plan_exhausted","invalid_state"])

    def test_success_at_last_action_and_initial_success(self):
        l=core.FirstAttempt(2,2)
        l.update(0,[.95,.9],[False]*2)
        l.update(2,[.1,.95],[False]*2,exhausted=[False,True])
        self.assertEqual(l.step.tolist(),[0,2])
        self.assertTrue(all(r["success"] for r in l.rows()))

    def test_shadow_and_age_follow_object_identity(self):
        perm=np.array([2,0,1,3,4,5,6,7,8,9])
        order=core.token_order(perm)
        world=np.arange(11*8,dtype=np.float32).reshape(11,8)+1
        teacher=np.r_[world[order].reshape(-1),[.5,0,.2,.2,.2,.2]]
        vis=np.ones(11,np.float32);vis[1]=0
        age=np.zeros(11,np.float32);age[1]=7
        obs=core.aligned_student_obs(so,teacher,np.zeros((11,2)),vis,age,perm,
                                     -1,[[.5,0]],0,[],[np.zeros((11,2))])
        tokens=obs[:110].reshape(11,10)
        slot=int(np.where(order==1)[0][0])
        self.assertTrue((tokens[slot,:8]==0).all())
        self.assertEqual(tokens[slot,8],0)
        self.assertAlmostEqual(float(tokens[slot,9]),.7)
        self.assertTrue(np.array_equal(tokens[1,:8],world[3]))

    def test_total_dropout_removes_all_current_geometry(self):
        v=so.apply_random_occlusion(np.ones(11),np.random.default_rng(0),1.)
        obs=core.aligned_student_obs(so,np.ones(94),np.ones((11,2)),v,np.zeros(11),
            np.arange(10),-1,[[.5,0]],0,[],[np.zeros((11,2))])
        self.assertTrue((obs[:110].reshape(11,10)[:,:9]==0).all())

    def test_random_streams_are_scene_order_invariant(self):
        streams={h:np.random.default_rng(core.scene_seed(h,7,"pose")).random(10) for h in ["a","b"]}
        for h in ["b","a"]:
            np.testing.assert_array_equal(streams[h],np.random.default_rng(core.scene_seed(h,7,"pose")).random(10))
        self.assertNotEqual(core.scene_seed("a",7,"pose"),core.scene_seed("a",7,"visibility"))

    def test_manifest_rejects_duplicate_and_modified_files(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);a=d/"a.txt";b=d/"b.txt";a.write_text("one");b.write_text("one")
            manifest=d/"m.json"
            manifest.write_text(json.dumps(dict(scenes=[dict(path=str(p),sha256=core.sha256(p)) for p in [a,b]])))
            with self.assertRaisesRegex(ValueError,"Duplicate"): core.read_manifest(manifest)
            b.write_text("two")
            with self.assertRaisesRegex(ValueError,"changed"): core.read_manifest(manifest)


if __name__=="__main__": unittest.main()
