from collections import Counter

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F

nltk.download("punkt_tab")


class W2V(nn.Module):
    def __init__(self, vocabulary_size: int, emb_dim: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.emb_dim = emb_dim

        # Encoder
        self.emb = nn.Embedding(vocabulary_size, emb_dim)
        torch.nn.init.uniform_(self.emb.weight, -0.5, 0.5)

        # Decoder
        self.unemb = nn.Embedding(emb_dim, vocabulary_size)
        torch.nn.init.uniform_(self.unemb.weight, -0.5, 0.5)

    def forward(
        self,
        center_words: torch.Tensor,
        target_words: torch.Tensor,
        negative_words: torch.Tensor,
    ):
        """
        Args:
            center_words: torch.Tensor shape=(N,)
            target_words: torch.Tensor shape=(N,)
            negative_words: torch.Tensor shape=(N, K)
        """
        emb_center = self.emb(center_words)  # (N, D)
        emb_target = self.emb(target_words)  # (N, D)
        emb_negative = self.emb(negative_words)  # (N, K, D)

        # (N, D) * (N, D) -> (N, D) -> sum -> (N,)
        positive_score = torch.sum(emb_center * emb_target, axis=1)

        # (N, 1, D) * (N, K, D) -> (N, K, D) -> sum -> (N, K)
        negative_score = torch.sum(emb_center.unsqueeze(1) * emb_negative, axis=2)

        # BCE
        # L = -[ylog(p) + (1-y)log(1-p)]
        # L(y=1) = -logp, L(y=0) = -log(1-p)
        # also using 1 - sigmoid(x) = sigmoid(-x) identity in negative_loss!
        positive_loss = -F.logsigmoid(positive_score)
        negative_loss = -F.logsigmoid(-negative_score)

        loss = (positive_loss + negative_loss.sum(axis=1)).mean()
        return loss


class W2VDataset(Dataset):
    def __init__(self, filepath: str, window_size: int = 2, k: int = 5):
        self.window_size = window_size
        self.k = k

        # read whole text
        with open(filepath, "r") as f:
            text = f.read().lower()

        # split into sentences
        sentences = nltk.sent_tokenize(text)

        # split into words and remove unnecessary characters
        sentences = list(map(nltk.word_tokenize, sentences))

        # build vocabulary
        all_words = [word for sent in sentences for word in sent]
        self.words_set = set(all_words)
        words_counted = Counter(all_words)

        self.word_to_id = {}
        self.id_to_word = {}

        for i, (word, _) in enumerate(words_counted.most_common()):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

        self.vocabulary_size = len(self.word_to_id)

        # generate training pairs
        self.data = []

        sentences_ids = [[self.word_to_id[w] for w in s] for s in sentences]

        for sentence in sentences_ids:
            for i, center_word in enumerate(sentence):
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size)

                for j in range(start, end + 1):
                    if i != j:
                        target_word = sentence[j]
                        self.data.append((center_word, target_word))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center, target = self.data[idx]
        negatives = torch.randint(low=0, high=self.vocabulary_size, size=(self.k,))
        return torch.tensor(center), torch.tensor(target), negatives

    def __len__(self):
        return len(self.data)


def train(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    lr=1e-3,
    batch_size=32,
) -> torch.nn.Module:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in tqdm(range(epochs)):
        for batch in dataloader:
            loss = model.forward(*batch)
            model.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate(model: nn.Module, dataset: Dataset, batch_size=32) -> None:
    dataloader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            loss = model.forward(*batch)
            total_loss += loss

    print("Total loss:", total_loss)
