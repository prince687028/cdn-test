import sys
sys.path.append('./')

import clip
import torch
from datasets.hico_text_label import hico_text_label


def init_classifier_with_CLIP(hoi_text_label):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_inputs = torch.cat([clip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])
    # print(text_inputs.shape)
    clip_model_param = 'ViT-B/32'
    
    clip_model, preprocess = clip.load(clip_model_param, device=device)
    
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_inputs.to(device))

        v_linear_proj_weight = clip_model.visual.proj.detach()

    del clip_model

    return text_embedding.float(),  v_linear_proj_weight.float()

clip_label, _ = init_classifier_with_CLIP(hico_text_label)

print(clip_label.shape)
print(clip_label[0].shape)
# print(clip_label[0])