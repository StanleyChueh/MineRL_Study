# MineRL_Learning
A GitHub repository documenting my MineRL learning journey and common issues related to installation and usage.

THIS STUDY AIM TO LET MINECRAFT AGENT FIND A TREE AND CUT IT BY THEIR OWN.
## MineRL setup
![Untitled ‑ Made with FlexClip (57)](https://github.com/user-attachments/assets/8677e18c-8402-453b-9325-6f39d6d39f73)

One of the human-playing data for model inferencing(Cut tree task)

## MineRL Usage
```
cd MineRL_Learning/vpt
python -m venv venv
source venv/bin/activate
```
## Collect data
Record human-playing video(mp4), press q to quit and save as jsonl
```
python main.py
```
## Model Fine-tuning
```
cd ..
cd Video-Pre-Training/
export PYTHONPATH=$PYTHONPATH:~/Video-Pre-Training
```
Fine-tuning model based on mp4 and jsonl
```
python behavioural_cloning.py     --data-dir stanley_cut_tree/data     --in-model foundation-model-1x.model     --in-weights foundation-model-1x.weights     --out-weights stanley_cut_tree/fine_tuned_weights.pth
```
## Model Inference
```
python inference_model.py --model /home/stanley/Video-Pre-Training/foundation-model-1x.model --weights /home/stanley/Video-Pre-Training/stanley_cut_tree/fine_tuned_weights.pth
```

## Reference
1.MineRL official: https://github.com/minerllabs/minerl.git

2.OpenAI Video-Pre-Training: https://github.com/openai/Video-Pre-Training.git

3.vpt: https://github.com/Infatoshi/vpt.git
