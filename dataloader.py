import pickle

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class IEMOCAPDataset(Dataset):
    """Dataset wrapper for pre-extracted IEMOCAP multimodal features."""

    def __init__(self, path, windows, train=True):
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.roberta2,
            self.roberta3,
            self.roberta4,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = pickle.load(open(path, "rb"), encoding="latin1")

        self.keys = list(self.trainVid if train else self.testVid)
        self.windows = windows
        self.len = len(self.keys)

    def get_semantic_adj(self, data):
        """Build the full relation adjacency matrix within the context window."""
        max_len = max(len(item[3]) for item in data)
        semantic_adj = []

        for item in data:
            speakers = item[3].tolist()
            adj = torch.zeros(max_len, max_len, dtype=torch.long)

            for i in range(len(speakers)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0:
                        continue
                    if j == len(speakers):
                        break

                    if speakers[i] == speakers[j]:
                        if i == j:
                            adj[i, j] = 1
                        elif i < j:
                            adj[i, j] = 2
                        else:
                            adj[i, j] = 3
                    else:
                        if i < j:
                            adj[i, j] = 4
                        elif i > j:
                            adj[i, j] = 5

            semantic_adj.append(adj)

        return torch.stack(semantic_adj)

    def getSelf_semantic_adj(self, data):
        """Build the intra-speaker relation adjacency matrix."""
        max_len = max(len(item[3]) for item in data)
        self_semantic_adj = []

        for item in data:
            speakers = item[3].tolist()
            adj = torch.zeros(max_len, max_len, dtype=torch.long)

            for i in range(len(speakers)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0:
                        continue
                    if j == len(speakers):
                        break

                    if speakers[i] == speakers[j]:
                        if i == j:
                            adj[i, j] = 1
                        elif i > j:
                            adj[i, j] = 2
                        else:
                            adj[i, j] = 3

            self_semantic_adj.append(adj)

        return torch.stack(self_semantic_adj)

    def getCross_semantic_adj(self, data):
        """Build the inter-speaker relation adjacency matrix."""
        max_len = max(len(item[3]) for item in data)
        cross_semantic_adj = []

        for item in data:
            speakers = item[3].tolist()
            adj = torch.zeros(max_len, max_len, dtype=torch.long)

            for i in range(len(speakers)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0:
                        continue
                    if j == len(speakers):
                        break

                    if speakers[i] == speakers[j]:
                        if i == j:
                            adj[i, j] = 1
                    else:
                        if i < j:
                            adj[i, j] = 4
                        elif i > j:
                            adj[i, j] = 5

            cross_semantic_adj.append(adj)

        return torch.stack(cross_semantic_adj)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(
                [[1, 0] if speaker == "M" else [0, 1] for speaker in self.videoSpeakers[vid]]
            ),
            torch.FloatTensor([1] * len(self.videoLabels[vid])),
            torch.LongTensor(self.videoLabels[vid]),
        )

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        batch = pd.DataFrame(data)

        self_semantic_adj = self.getSelf_semantic_adj(data)
        cross_semantic_adj = self.getCross_semantic_adj(data)
        semantic_adj = self.get_semantic_adj(data)

        collated = [
            pad_sequence(batch[i])
            if i < 4
            else pad_sequence(batch[i], batch_first=True)
            if i < 6
            else batch[i].tolist()
            for i in batch
        ]
        collated.append(self_semantic_adj.long())
        collated.append(cross_semantic_adj.long())
        collated.append(semantic_adj.long())
        return collated


class MELDDataset(Dataset):
    """Dataset wrapper for pre-extracted MELD multimodal features."""

    def __init__(self, path, windows, train=True):
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.roberta2,
            self.roberta3,
            self.roberta4,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
            _,
        ) = pickle.load(open(path, "rb"), encoding="latin1")

        self.keys = list(self.trainVid if train else self.testVid)
        self.windows = windows
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.IntTensor(self.videoSpeakers[vid]),
            torch.FloatTensor([1] * len(self.videoLabels[vid])),
            torch.LongTensor(self.videoLabels[vid]),
        )

    def __len__(self):
        return self.len

    def get_semantic_adj(self, data):
        """
        Build the full relation adjacency matrix.

        This preserves the original MELD preprocessing logic, where the full
        semantic graph is constructed over the complete dialogue.
        """
        max_len = max(len(item[3]) for item in data)
        semantic_adj = []

        for item in data:
            speakers = item[3].tolist()
            adj = torch.zeros(max_len, max_len, dtype=torch.long)

            for i in range(len(speakers)):
                for j in range(len(speakers)):
                    if speakers[i] == speakers[j]:
                        if i == j:
                            adj[i, j] = 1
                        elif i < j:
                            adj[i, j] = 2
                        else:
                            adj[i, j] = 3
                    else:
                        if i < j:
                            adj[i, j] = 4
                        elif i > j:
                            adj[i, j] = 5

            semantic_adj.append(adj)

        return torch.stack(semantic_adj)

    def getSelf_semantic_adj(self, data):
        """Build the intra-speaker relation adjacency matrix within the context window."""
        max_len = max(len(item[3]) for item in data)
        self_semantic_adj = []

        for item in data:
            speakers = item[3].tolist()
            adj = torch.zeros(max_len, max_len, dtype=torch.long)

            for i in range(len(speakers)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0:
                        continue
                    if j == len(speakers):
                        break

                    if speakers[i] == speakers[j]:
                        if i == j:
                            adj[i, j] = 1
                        elif i > j:
                            adj[i, j] = 2
                        else:
                            adj[i, j] = 3

            self_semantic_adj.append(adj)

        return torch.stack(self_semantic_adj)

    def getCross_semantic_adj(self, data):
        """Build the inter-speaker relation adjacency matrix within the context window."""
        max_len = max(len(item[3]) for item in data)
        cross_semantic_adj = []

        for item in data:
            speakers = item[3].tolist()
            adj = torch.zeros(max_len, max_len, dtype=torch.long)

            for i in range(len(speakers)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0:
                        continue
                    if j == len(speakers):
                        break

                    if speakers[i] == speakers[j]:
                        if i == j:
                            adj[i, j] = 1
                    else:
                        if i < j:
                            adj[i, j] = 4
                        elif i > j:
                            adj[i, j] = 5

            cross_semantic_adj.append(adj)

        return torch.stack(cross_semantic_adj)

    def return_labels(self):
        labels = []
        for key in self.keys:
            labels.extend(self.videoLabels[key])
        return labels

    def collate_fn(self, data):
        batch = pd.DataFrame(data)

        self_semantic_adj = self.getSelf_semantic_adj(data)
        cross_semantic_adj = self.getCross_semantic_adj(data)
        semantic_adj = self.get_semantic_adj(data)

        collated = [
            pad_sequence(batch[i])
            if i < 4
            else pad_sequence(batch[i], batch_first=True)
            if i < 6
            else batch[i].tolist()
            for i in batch
        ]
        collated.append(self_semantic_adj.long())
        collated.append(cross_semantic_adj.long())
        collated.append(semantic_adj.long())
        return collated
