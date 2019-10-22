import numpy as np
from PIL import ImageEnhance


transformtypedict=dict(Brightness=ImageEnhance.Brightness, 
                       Contrast=ImageEnhance.Contrast, 
                       Sharpness=ImageEnhance.Sharpness, 
                       Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):  
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = np.random.uniform(0, 1, len(self.transforms)) 

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1   
            out = transformer(out).enhance(r)

        return out
