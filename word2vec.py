from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pl_preprocess import words_from_file


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -torch.sum(targets * torch.log(logits), dim=-1).mean()

def cross_entropy_one_class_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -torch.log(logits[target]).mean()

def softmax(logits: torch.Tensor) -> torch.Tensor:
    logits_norm = logits - torch.max(logits, dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits_norm)
    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    return probs

class Word2Vec(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, x):
        return self.linear(self.embedding(x))


def train(
    model: Word2Vec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 0.01,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        total_loss = 0.0

        for center, targets in train_loader:
            center = center.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(center)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")


@torch.inference_mode()
def evaluate(model: Word2Vec, dataloader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    for center, targets in dataloader:
        center = center.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(center)
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, targets)
        total_loss += loss.item()
    return total_loss / len(dataloader)


def _read_words_lower(filepath: str) -> list[str]:
    words = words_from_file(filepath)
    if not words:
        raise ValueError(f"Brak słów po tokenizacji: {filepath!r}")
    return words


def _build_vocab(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    words_counted = Counter(words)
    word_to_id: dict[str, int] = {}
    id_to_word: dict[int, str] = {}
    for i, (word, _) in enumerate(words_counted.most_common()):
        word_to_id[word] = i
        id_to_word[i] = word
    return word_to_id, id_to_word


def _pairs_in_token_span(
    ids: list[int],
    span_start: int,
    span_end_exclusive: int,
    window_size: int,
) -> list[tuple[int, int]]:
    """Pary (center, context) tylko w [span_start, span_end_exclusive); okno nie wychyla się poza span."""
    n = len(ids)
    lo_b = max(0, span_start)
    hi_b = min(n, span_end_exclusive)
    pairs: list[tuple[int, int]] = []
    w = window_size
    for i in range(lo_b, hi_b):
        lo = max(lo_b, i - w)
        hi = min(hi_b, i + w + 1)
        for j in range(lo, hi):
            if j != i:
                pairs.append((ids[i], ids[j]))
    return pairs


def make_train_val_w2v_datasets(
    filepath: str,
    window_size: int = 2,
    train_fraction: float = 0.9,
) -> tuple["W2VDataset", "W2VDataset"]:
    """
    Chronologiczny podział strumienia tokenów: train = początek korpusu, val = koniec.
    Słownik z całego tekstu (wspólne embeddingi); pary w val nie korzystają z kontekstu z train.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction musi być w (0, 1).")
    words = _read_words_lower(filepath)
    word_to_id, id_to_word = _build_vocab(words)
    ids = [word_to_id[w] for w in words]
    split = int(len(ids) * train_fraction)
    w = window_size
    if split <= w or len(ids) - split <= w:
        raise ValueError(
            f"Za mały segment train lub val (len={len(ids)}, split={split}, window={w}). "
            "Zmniejsz window_size lub train_fraction."
        )
    train_pairs = _pairs_in_token_span(ids, 0, split, window_size)
    val_pairs = _pairs_in_token_span(ids, split, len(ids), window_size)
    if not train_pairs or not val_pairs:
        raise ValueError("Pusty zbiór par train lub val — zwiększ korpus lub zmień parametry podziału.")
    meta = (word_to_id, id_to_word, len(word_to_id), window_size)
    return W2VDataset(train_pairs, meta), W2VDataset(val_pairs, meta)


class W2VDataset(Dataset):
    """Skip-gram: lista par (center_id, ctx_id) + wspólny słownik. Pełny korpus: `make_train_val_w2v_datasets`."""

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        meta: tuple[dict[str, int], dict[int, str], int, int],
    ):
        word_to_id, id_to_word, vocab_size, window_size = meta
        self.pairs = pairs
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.words_set = set(word_to_id)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        center_id, ctx_id = self.pairs[idx]
        center = torch.tensor(center_id, dtype=torch.long)
        targets = nn.functional.one_hot(
            torch.tensor(ctx_id, dtype=torch.long),
            self.vocab_size,
        ).float()
        return center, targets


@torch.no_grad()
def closest_words_analogy(
    model: Word2Vec,
    dataset: W2VDataset,
    w1: str,
    minus: str,
    plus: str,
    top_k: int = 15,
) -> list[tuple[str, float]]:
    for w in (w1, minus, plus):
        if w not in dataset.word_to_id:
            raise KeyError(f"Słowo {w!r} nie występuje w korpusie (brak w słowniku).")

    E = model.embedding.weight
    i1 = dataset.word_to_id[w1]
    im = dataset.word_to_id[minus]
    ip = dataset.word_to_id[plus]
    v = E[i1] - E[im] + E[ip]
    v = v / (v.norm() + 1e-8)

    norms = E.norm(dim=1, keepdim=True).clamp(min=1e-8)
    En = E / norms
    sims = En @ v

    exclude = {i1, im, ip}
    sims_masked = sims.clone()
    sims_masked[list(exclude)] = float("-inf")

    k = min(top_k, dataset.vocab_size - len(exclude))
    vals, idx = torch.topk(sims_masked, k=k)
    return [(dataset.id_to_word[int(i)], float(vals[j])) for j, i in enumerate(idx)]


if __name__ == "__main__":
    print(f"DEVICE={DEVICE}")
    WINDOW_SIZE = 2
    EMBEDDING_DIM = 256
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 0.01
    TRAIN_FRACTION = 0.9

    train_ds, val_ds = make_train_val_w2v_datasets(
        "data/pan-tadeusz.txt",
        window_size=WINDOW_SIZE,
        train_fraction=TRAIN_FRACTION,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = Word2Vec(train_ds.vocab_size, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    print(
        f"Pary: train={len(train_ds)}, val={len(val_ds)}, vocab={train_ds.vocab_size}, "
        f"train_frac={TRAIN_FRACTION}",
    )
    print(
        "Initial:",
        f"train loss={evaluate(model, train_loader):.4f},",
        f"val loss={evaluate(model, val_loader):.4f}",
    )
    train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

    for word, score in closest_words_analogy(model, train_ds, "król", "mężczyzna", "kobieta"):
        print(f"{word!r}: {score:.4f}")