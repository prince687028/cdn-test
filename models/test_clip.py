from calendar import c
import sys
sys.path.append('./')

import clip
import torch
from datasets.hico_text_label import hico_text_label

def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min = 1e-4, max = 1 - 1e-4)
    return y

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

device = "cuda" if torch.cuda.is_available() else "cpu"

bs = 2

a = torch.ones(bs,512).to(device)

a = a / 512

res = torch.matmul(a, clip_label.permute(1, 0))

print(res.shape)

# res below is out_clip_logits

res_k, res_idx = torch.topk(res, 16, dim = 1)

# print(res_k)

res_k = _sigmoid(res_k)

res_k = torch.softmax(res_k, dim = 1)

print(res_k)

# print(res_idx)

res_embedding = clip_label[res_idx,:]

# print(res_embedding[0][3].sum())
# print(clip_label[97].sum())

print(res_embedding.shape)
# print(res_embedding)

res_k1 = res_k.unsqueeze(2)
print(res_k1.shape)

ans = torch.mul(res_k1, res_embedding)

print(ans.shape)

print(ans[0][15].sum())
print(res_embedding[0][15].sum() * res_k[0][15])

print(ans.sum(dim = 1).shape)
