import numpy as np
import torch

class DBNormalize:
    def __init__(self,mean:list):
        self.mean:np.ndarray = np.array(mean)

    def build(self,data:dict,retype_polygon=True) -> dict:
        assert 'image' in data
        image:np.ndarray = data['image'].astype(np.float64)
        image -= self.mean
        image /= 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        data['image'] = image
        if retype_polygon:
            data['polygon'] = np.array(data['polygon'],np.float32)
        return data