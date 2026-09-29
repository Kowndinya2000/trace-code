"""CPU-only integrity and inference checks for the selected-ep210 release.

Run from the repository root with the project's Python environment. This does
not construct a simulator, connect to a robot, or execute motion commands.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from isaacgymenvs.learning.student_net import StudentNet
from isaacgymenvs.learning.student_ablate import obs_mask, strip_recurrence
from isaacgymenvs.open_loop.evaluation_core import primitive_vectors


def load_student(path):
    ck = torch.load(path, map_location="cpu")
    sd = ck["model"]
    assert ck["obs_dim"] == 166 and not ck["smoke"]
    hidden = (sd["gru.weight_ih_l0"].shape[0] // 3
              if "gru.weight_ih_l0" in sd else sd["pi.weight"].shape[1])
    net = StudentNet(n_actions=sd["pi.weight"].shape[0],
                     embed=sd["embed.0.weight"].shape[0], gru_hidden=hidden,
                     termination_head='stop.weight' in sd)
    if ck["ablate"] == "no_gru":
        net = strip_recurrence(net)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net, obs_mask(ck["ablate"], "cpu"), ck


def check_teacher(path, params, epoch):
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning.token_set_builder import TokenSetBuilder
    model_builder.register_network("token_set", lambda **kwargs: TokenSetBuilder())
    ck = torch.load(path, map_location="cpu")
    assert ck["epoch"] == epoch
    with contextlib.redirect_stdout(io.StringIO()):
        factory = model_builder.ModelBuilder().load(params)
        net = factory.build(dict(actions_num=16, input_shape=(94,),
            num_seqs=1, value_size=1, normalize_input=True, normalize_value=True))
    net.load_state_dict(ck["model"], strict=True)
    net.eval()
    state = net.a2c_network.get_default_rnn_state()
    with torch.no_grad():
        out = net(dict(is_train=False, prev_actions=None,
                       obs=torch.zeros(1, 94), rnn_states=state))
    assert torch.isfinite(out["logits"]).all()
    return dict(epoch=epoch, strict_state_dict=True, finite_cpu_forward=True)


def verify(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    # Check every payload (including configs and recorded inference probes)
    # before loading any serialized model.
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (bundle / name).resolve()
        if bundle.resolve() not in path.parents:
            raise ValueError("Checksum path escapes bundle")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    params = json.loads((bundle / "teacher_network.json").read_text())
    report = dict(valid=True, torch_version=torch.__version__, device="cpu",
                  robot_connected=False, checkpoints={})
    P = torch.from_numpy(primitive_vectors(0.04))
    with np.load(bundle / "validation/student_probes.npz", allow_pickle=False) as probes:
        for item in manifest["checkpoints"]:
            path = bundle / item["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            if item["kind"] == "teacher":
                result = check_teacher(path, params, item["epoch"])
            else:
                net, mask, ck = load_student(path)
                assert ck["head"] == item["head"] and ck["ablate"] == item["ablation"]
                x = torch.from_numpy(probes[item["id"] + "_obs"]).unsqueeze(1)
                if mask is not None:
                    x = x * mask
                h, steps = None, []
                with torch.no_grad():
                    whole, _, _ = net(x)
                    for obs in x:
                        y, _, h = net(obs.unsqueeze(0), h)
                        steps.append(y)
                stepwise = torch.cat(steps)
                torch.testing.assert_close(stepwise, whole, rtol=1e-4, atol=1e-6)
                assert torch.isfinite(stepwise).all()
                actions = (torch.cdist(stepwise[:, 0], P).argmin(-1)
                           if item["head"] == "xy" else stepwise[:, 0].argmax(-1))
                expected = torch.from_numpy(probes[item["id"] + "_actions"])
                assert torch.equal(actions, expected), (item["id"], actions, expected)
                result = dict(strict_state_dict=True, finite_cpu_forward=True,
                    recurrent_sequence_matches=True, recorded_actions_matched=len(expected))
            report["checkpoints"][item["id"]] = result
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = verify(args.bundle.resolve())
    payload = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(payload)
    print(payload)
