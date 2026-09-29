import json
from pathlib import Path
import unittest

import numpy as np
import torch

from isaacgymenvs.open_loop.teacher_policy import TeacherPolicy


import os

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(os.environ.get("TRACE_DATA", ROOT.parents[1] / "data")) / "checkpoints"


class TeacherPolicyTest(unittest.TestCase):
    @unittest.skipUnless((BUNDLE / "teacher_ep210.pth").exists(),
                         "teacher checkpoint not downloaded")
    def test_teacher_wrapper_is_exact_and_recurrent(self):
        teacher = TeacherPolicy(BUNDLE / "teacher_ep210.pth",
                                BUNDLE / "teacher_network.json", device="cpu")
        self.assertEqual(teacher.epoch, 210)
        rng = np.random.default_rng(7)
        sequence = rng.normal(size=(3, 94)).astype(np.float32)

        # Compare the wrapper against a direct forward through the exact model,
        # using an independently managed recurrent state.
        direct_state = tuple(s.clone() for s in teacher.states)
        for obs in sequence:
            action, logits = teacher.step(obs)
            with torch.no_grad():
                out = teacher.model(dict(is_train=False, prev_actions=None,
                                         obs=torch.from_numpy(obs).unsqueeze(0),
                                         rnn_states=direct_state))
            direct_state = tuple(s.detach() for s in out["rnn_states"])
            expected = out["logits"][0].detach().numpy()
            np.testing.assert_allclose(logits, expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(action, int(expected.argmax()))

        teacher.reset()
        action0, logits0 = teacher.step(sequence[0])
        teacher.reset()
        action1, logits1 = teacher.step(sequence[0])
        self.assertEqual(action0, action1)
        np.testing.assert_array_equal(logits0, logits1)

    @unittest.skipUnless((BUNDLE / "teacher_network.json").exists(),
                         "teacher network spec not downloaded")
    def test_selected_teacher_network_contract(self):
        params = json.loads((BUNDLE / "teacher_network.json").read_text())
        network = params["network"]
        self.assertEqual(network["name"], "token_set")
        self.assertEqual(network["token_set"], {
            "nTokens": 11, "tokenDim": 8, "embed": 64,
            "layers": 2, "studentCompat": True,
        })
        self.assertEqual(network["rnn"]["name"], "gru")
        self.assertIs(network["rnn"]["concat_input"], True)


if __name__ == "__main__":
    unittest.main()
