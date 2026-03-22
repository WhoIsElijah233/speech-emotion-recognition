# import torch
# from torch.utils.data import Dataset

# class EmotionDataset(Dataset):
#   def __init__(self, size = 100):
#     self.size = size

#   def __len__(self):
#     return self.size

#   def __getitem__(self, idx):
#     audio = torch.randn(16000)

#     label = torch.randint(0, 4, (1,)).item()

#     return audio, label

import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


class EmotionDataset(Dataset):

    def __init__(self, root="/content/drive/MyDrive/emotion_dataset/RAVDESS"):

        self.files = []

        for actor in os.listdir(root):

            if actor.startswith("Actor"):

                actor_path = os.path.join(root, actor)

                for file in os.listdir(actor_path):

                    if file.endswith(".wav"):

                        full_path = os.path.join(actor_path, file)

                        self.files.append(full_path)

        # 重采样
        self.resample = T.Resample(48000, 16000)

        # 频谱
        self.mel = T.MelSpectrogram(
            sample_rate=16000,
            n_mels=80
        )


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        waveform, sr = torchaudio.load(path)

        # stereo -> mono
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = self.resample(waveform)

        spec = self.mel(waveform)

        max_len = 300

        if spec.shape[2] < max_len:
            pad = max_len - spec.shape[2]
            spec = torch.nn.functional.pad(spec, (0, pad))
        else:
            spec = spec[:, :, :max_len]

        filename = os.path.basename(path)
        emotion = int(filename.split("-")[2]) - 1

        return spec, emotion
    
