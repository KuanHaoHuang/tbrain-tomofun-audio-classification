from PIL import Image
import json
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models, transforms
from model import TomoModel
from utils import extract_feature

print("predict.py starts intializing")
############### 前置作業 ###############
# 標籤與 id 對應
COLUMNS = "Barking,Howling,Crying,COSmoke,GlassBreaking,Other,Doorbell,Bird,Music_Instrument,Laugh_Shout_Scream"
id2label = {i: x for i, x in enumerate(COLUMNS.split(","))}
label2id = {v: k for k, v in id2label.items()}

# 讀取模型架構

# 讀取已訓練好的模型權重
model_weights_file_path = Path('./model_weights')
models = []
for f in model_weights_file_path.iterdir():
    if not f.name.endswith('.pkl'): continue
    model_ft = TomoModel()
    model_ft.load_state_dict(
            torch.load(f)
        )
    model_ft.eval()
    models.append(model_ft.to('cuda'))
#######################################
print(f"predict.py intialized! There are {len(models)} models.")

# 預測函數
def predict(wav):
    """ Predict your model result.

    @param:
        wav file (Path)
    @returns: (label, probability)
        label (int)
        probability (list of size=10)
    """
    entry = extract_feature(wav)
    values = entry["values"]
    values = torch.Tensor(values.reshape(-1, 128, 250))
    values = values.unsqueeze(0).to('cuda')  # 增加一個batch維度
    
    result_probs = None
    for i in range(len(models)):
        model_ft = models[i]
        outputs = model_ft(values)
        probs = torch.exp(outputs)
        if result_probs is None:
            result_probs = probs
        else:
            result_probs += probs
    result_probs /= len(models)
    _, pred = torch.max(result_probs, 1)

    # 模型最終預測結果
    label = pred.item()
    probability = result_probs.squeeze().tolist()
    probability = [round(x, 6) for x in probability]
    diff = round(1 - sum(probability), 6)
    probability[label] += diff

    return (label, probability)
