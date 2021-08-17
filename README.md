# T-brain 2021，Tomofun 狗音辨識 AI 百萬挑戰賽：Top 3% Solution
- 競賽名稱：Tomofun 狗音辨識 AI 百萬挑戰賽
- 競賽網址：https://tbrain.trendmicro.com.tw/Competitions/Details/15
- 競賽簡述：本次競賽提供居家環境中可能會發生的聲音，參賽者需用機器學習模型做出居家環境音的識別。

<img src="https://i.imgur.com/CCsE3c6.png" alt="demo" width="400"/>

## 參賽成績
- Team: StarRingChild
- Rank: 10th / 301 teams (Top 3%)
- AUC：0.934

## Git Repo 說明
- train_model.ipynb <a href="https://colab.research.google.com/github/KuanHaoHuang/tbrain-tomofun-audio-classification/blob/main/train_model.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>: 模型訓練
- app.py: Flask 模型部署

## [競賽心得部落格文章](https://haosquare.com/tbrain-tomofun-audio-classification/)

## 比賽技巧簡述
- 多通道的 Spectrogram 轉換，同時採用不同 Window Size
- DenseNet + Ensemble Model
- ImageNet 預訓練的 DenseNet 對音訊辨識任務的遷移學習

## Reference
[Rethinking CNN Models for Audio Classification](https://arxiv.org/abs/2007.11154)
  
