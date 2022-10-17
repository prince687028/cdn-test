import torch

from hico_text_label import hico_text_label

target = torch.zeros(len(hico_text_label))

print(target.shape)

hoi_target = (52, 4)

hico_text_label_keys = list(hico_text_label.keys())

# print(hico_text_label_key)

if hoi_target in hico_text_label_keys:
    idx = hico_text_label_keys.index(hoi_target)
    target[idx] = 1

print(target[:10])
