"""Causal image preprocessing: no hidden target centers or future observations."""
import cv2
import numpy as np

SIZE=112
META_DIM=26  # EEF xy, previous action(16), crop center(2), seen/age, t/L, L/H, drift xy
DEPTH_MEAN=.0019
DEPTH_STD=.0091


def crop(image, center, size=SIZE, fill=0):
    h,w=image.shape[:2]
    x,y=np.rint(center).astype(int)
    half=size//2
    padded=np.pad(image,((half,half),(half,half))+((0,0),)*(image.ndim-2),constant_values=fill)
    return padded[y:y+size,x:x+size].copy()


class TargetHistory:
    def __init__(self, initial_depth, initial_target_mask):
        ys,xs=np.nonzero(initial_target_mask)
        if not len(xs): raise ValueError('A clean initial target detection is required')
        self.center=np.array([xs.mean(),ys.mean()],np.float32)
        self.age=0
        self.shape=initial_depth.shape
        self.initial_target_pixels=len(xs)
        # Fixed table reference from the permitted clean initial observation.
        h,w=initial_depth.shape
        c0=max((h-224)//2,0)
        clean=np.where(np.isfinite(initial_depth),initial_depth,0)
        self.plane=float(clean[c0:c0+224,c0:c0+224].min())

    def encode(self, depth, target_mask, unavailable, eef, previous_action, t, nominal_xy):
        depth=np.asarray(depth,np.float32)
        missing=np.asarray(unavailable,bool)|~np.isfinite(depth)
        target=np.asarray(target_mask,bool)&~missing
        ys,xs=np.nonzero(target)
        seen=bool(len(xs))
        if seen:
            self.center[:]=[xs.mean(),ys.mean()]
            self.age=0
        else: self.age+=1
        height=np.where(missing,0,np.clip(depth-self.plane,0,.25)).astype(np.float32)
        local=crop(height,self.center)
        global_image=cv2.resize(height,(SIZE,SIZE),interpolation=cv2.INTER_AREA)
        local_masks=np.stack([crop(target.astype(np.uint8),self.center),crop(missing.astype(np.uint8),self.center,fill=1)])
        global_masks=np.stack([cv2.resize(target.astype(np.uint8),(SIZE,SIZE),interpolation=cv2.INTER_NEAREST),
                              cv2.resize(missing.astype(np.uint8),(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)])
        images=np.stack([local,global_image])
        masks=np.stack([local_masks,global_masks])
        length=max(len(nominal_xy)-1,0)
        meta=np.zeros(META_DIM,np.float32)
        meta[:2]=eef
        if previous_action>=0:meta[2+previous_action]=1
        meta[18:20]=self.center/np.array([depth.shape[1],depth.shape[0]])
        meta[20:24]=[seen,min(self.age,120)/120.,t/max(length,1),length/120.]
        if len(nominal_xy):meta[24:]=np.asarray(nominal_xy[min(t,len(nominal_xy)-1)])-eef
        return images,masks,meta


def augment_missing(mask,center,rng,t,blackout_start,blackout_length,target_hide_probability=.2):
    """Rectangular sensor failures are independent of hidden object geometry."""
    result=np.asarray(mask,bool).copy()
    if t and blackout_start<=t<blackout_start+blackout_length:
        result[:]=True
    elif t and rng.random()<target_hide_probability:
        h,w=result.shape;x,y=np.rint(center).astype(int)
        half=int(rng.integers(20,51))
        result[max(0,y-half):min(h,y+half),max(0,x-half):min(w,x+half)]=True
    return result
