"""Causal workspace-aligned depth memory from the existing sensor contract.

Uses only the recorded local crop, coarse overview, explicit missing mask and
last-detected crop center. In particular it never uses labels, true object
states, a clear *current* render, or observations from another masking stream.
The initial clean overview initializes a coarse map; observed local pixels
replace it at their native resolution. Unobserved pixels retain prior depth.
"""
import cv2
import numpy as np
from isaacgymenvs.open_loop.grasp_image_obs import crop, SIZE

CANVAS_SIZE = 320


class CausalCropMemory:
    def __init__(self):
        self.canvas = None
        self.target_canvas = None

    def observe(self, images, masks, meta):
        if images.shape != (2,SIZE,SIZE) or masks.shape != (2,2,SIZE,SIZE):
            raise ValueError('Wrong depth-memory input shape')
        if self.canvas is None:
            if not meta[20]:
                raise ValueError('Depth memory requires the permitted clean initial view')
            self.canvas = cv2.resize(images[1].astype(np.float32),
                (CANVAS_SIZE,CANVAS_SIZE),interpolation=cv2.INTER_NEAREST)
            self.target_canvas = cv2.resize(masks[1,0].astype(np.uint8),
                (CANVAS_SIZE,CANVAS_SIZE),interpolation=cv2.INTER_NEAREST)
        center = np.rint(np.asarray(meta[18:20])*CANVAS_SIZE).astype(int)
        x,y = center-SIZE//2
        left,top = max(x,0),max(y,0)
        right,bottom = min(x+SIZE,CANVAS_SIZE),min(y+SIZE,CANVAS_SIZE)
        if right>left and bottom>top:
            sx,sy = left-x,top-y
            patch = images[0,sy:sy+bottom-top,sx:sx+right-left]
            observed = ~masks[0,1,sy:sy+bottom-top,sx:sx+right-left].astype(bool)
            observed &= np.isfinite(patch)
            target = self.canvas[top:bottom,left:right]
            np.copyto(target,patch,where=observed)
            target_patch = masks[0,0,sy:sy+bottom-top,sx:sx+right-left]
            remembered_target = self.target_canvas[top:bottom,left:right]
            np.copyto(remembered_target,target_patch,where=observed)
        completed = crop(self.canvas,center)
        completed_target = crop(self.target_canvas,center)
        result = np.stack([images[0],completed]).astype(images.dtype)
        # Availability remains explicit even when a pixel has remembered depth.
        result_masks = np.stack([masks[0],masks[0]])
        # The second stream explicitly identifies causally remembered target
        # pixels.  Its unknown mask remains the current robot/sensor mask, so
        # memory is never confused with a fresh observation.
        result_masks[1,0] = completed_target
        return result,result_masks


def transform_sequence(images,masks,meta):
    # An already-retired step-0 OOW/invalid scene is deliberately archived as
    # one neutral negative frame with no invented target.  It has no temporal
    # memory to construct.  Live trajectories and multi-frame histories still
    # require the permitted clean initial target view.
    if len(images)==1 and not meta[0,20]:
        neutral=(not np.any(images[0]) and not np.any(masks[0,:,0])
                 and np.all(masks[0,:,1]))
        if not neutral:
            raise ValueError('Missing initial target is not a neutral terminal record')
        return images.copy(),masks.copy()
    memory = CausalCropMemory()
    result = [memory.observe(i,m,s) for i,m,s in zip(images,masks,meta)]
    return np.stack([r[0] for r in result]),np.stack([r[1] for r in result])
