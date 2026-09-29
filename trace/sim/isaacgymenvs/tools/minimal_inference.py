"""Bound frozen grasp-classifier inference memory without reducing PPO scale."""
import json
from pathlib import Path
import torch


class ChunkedGraspInference(torch.nn.Module):
    def __init__(self,model,report_path,batch_size=64):
        super().__init__()
        if model.training: raise ValueError('Grasp classifier must be frozen in eval mode')
        self.model=model;self.batch_size=batch_size;self.report_path=Path(report_path)
        self.checked=False
        self.eval()

    @torch.no_grad()
    def forward(self,x):
        if self.model.training: raise ValueError('Cannot chunk a training-mode classifier')
        if not self.checked:
            # Compare two inference batch sizes on actual scene crops before
            # accepting the memory adapter. This tests scores AND the stop label.
            probe=x[:2*self.batch_size]
            original=self.model(probe)
            adapted=torch.cat([self.model(p) for p in probe.split(self.batch_size)],dim=0)
            torch.testing.assert_close(adapted,original,rtol=0,atol=1e-5)
            torch.testing.assert_close(adapted.sigmoid()>.9,original.sigmoid()>.9)
            self.report_path.write_text(json.dumps(dict(valid=True,batch_size=self.batch_size,
                probe_crops=len(probe),max_logit_error=float((adapted-original).abs().max()),
                threshold_labels_match=True,model_training=False),indent=2)+'\n')
            self.checked=True
        return torch.cat([self.model(p) for p in x.split(self.batch_size)],dim=0)
