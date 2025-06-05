import json
import sys
from pathlib import Path

import hydra
import torch

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel


def predicts_to_annotation(imgfile: Path, width, height, predicts) -> dict:
    anno = {
        "file_name": imgfile.with_suffix('.tif').name,
        "width": width,
        "height": height,
        "annotations": [
            {
                'class': 'vacant_lot',
                'bbox': [int(pred[1]), int(pred[2]), int(pred[3] - pred[1]), int(pred[4] - pred[2])]
            }
        for pred in predicts[0]] 
    }
    return anno
    

@hydra.main(config_path="../cfg/yolo", config_name="config", version_base=None)
def main(cfg: Config):
    model = InferenceModel(cfg)
    model.setup(None)
    
    save_path = Path(cfg.out_path) / 'inference' / cfg.name
    save_path.mkdir(parents=True, exist_ok=True)
    
    annotations = []
    
    for imgfile, batch in model.predict_dataloader():
        with torch.no_grad():
            img, predicts = model.predict_step(batch, 0)
        
        save_image_path = save_path / imgfile.name
        # img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
        
        annotations.append(predicts_to_annotation(imgfile, img.width, img.height, predicts))
        
    with open(save_path / 'bbox.json', 'w') as f:
        json.dump({'images': annotations}, f, indent=4)
        print(f"💾 Saved annotations at {save_path / 'bbox.json'}")
    

if __name__ == "__main__":
    main()