import argparse
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ---------------------------
# 数据加载与预处理
# ---------------------------


def read_parallel_corpus(src_path: Path, tgt_path: Path, max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
    def _read_file(path: Path) -> List[str]:
        if path.suffix.lower() == ".xml":
            import xml.etree.ElementTree as ET

            texts: List[str] = []
            tree = ET.parse(path)
            root = tree.getroot()
            for seg in root.iterfind(".//seg"):
                text = (seg.text or "").strip()
                if text:
                    texts.append(text)
            return texts
        else:
            lines: List[str] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("<"):
                        continue
                    lines.append(line)
            return lines

    src_lines = _read_file(src_path)
    tgt_lines = _read_file(tgt_path)

    pairs: List[Tuple[str, str]] = []
    for idx, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
        pairs.append((src_line, tgt_line))
        if max_samples is not None and idx + 1 >= max_samples:
            break
    return pairs


def train_sentencepiece_model(
    texts: Sequence[str],
    model_path: Path,
    vocab_size: int,
    character_coverage: float = 1.0,
    retrain: bool = False,
) -> None:
    if model_path.exists() and not retrain:
        return
    if retrain:
        model_path.unlink(missing_ok=True)
        model_vocab_path = model_path.with_suffix(".vocab")
        model_vocab_path.unlink(missing_ok=True)
    model_prefix = model_path.with_suffix("")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        for line in texts:
            line = line.strip()
            if not line:
                continue
            tmp.write(line.lower() + "\n")
        tmp_path = tmp.name
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="unigram",
        pad_id=-1,
        bos_id=-1,
        eos_id=-1,
        hard_vocab_limit=False,
    )
    Path(tmp_path).unlink(missing_ok=True)


SPECIAL_TOKENS = {
    "pad": "<pad>",
    "bos": "<bos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


class Vocab:
    def __init__(self, counter: Dict[str, int], max_size: int = 32000, min_freq: int = 1) -> None:
        most_common = sorted(
            [item for item in counter.items() if item[1] >= min_freq],
            key=lambda x: (-x[1], x[0]),
        )
        words = [token for token, _ in most_common[: max_size - len(SPECIAL_TOKENS)]]

        self.itos: List[str] = [
            SPECIAL_TOKENS["pad"],
            SPECIAL_TOKENS["bos"],
            SPECIAL_TOKENS["eos"],
            SPECIAL_TOKENS["unk"],
        ] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_token(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[SPECIAL_TOKENS["unk"]])

    def lookup_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.lookup_token(tok) for tok in tokens]

    @property
    def pad_idx(self) -> int:
        return self.stoi[SPECIAL_TOKENS["pad"]]

    @property
    def bos_idx(self) -> int:
        return self.stoi[SPECIAL_TOKENS["bos"]]

    @property
    def eos_idx(self) -> int:
        return self.stoi[SPECIAL_TOKENS["eos"]]

    @classmethod
    def from_itos(cls, itos: Sequence[str]) -> "Vocab":
        obj = cls.__new__(cls)
        obj.itos = list(itos)
        obj.stoi = {w: i for i, w in enumerate(obj.itos)}
        return obj


def sentencepiece_tokenizer(sp: spm.SentencePieceProcessor) -> Callable[[str], Sequence[str]]:
    def _tokenize(text: str) -> List[str]:
        return sp.encode(text.lower(), out_type=str)

    return _tokenize


def build_vocab(
    sentences: Iterable[Sequence[str]],
    max_size: int = 32000,
    min_freq: int = 2,
) -> Vocab:
    counter: Dict[str, int] = {}
    for seq in sentences:
        for token in seq:
            counter[token] = counter.get(token, 0) + 1
    return Vocab(counter, max_size=max_size, min_freq=min_freq)


class TranslationDataset(Dataset[Tuple[List[int], List[int]]]):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        src_tokenizer: Callable[[str], Sequence[str]],
        tgt_tokenizer: Callable[[str], Sequence[str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_length: Optional[int] = None,
    ) -> None:
        self.examples: List[Tuple[List[int], List[int]]] = []
        for src_text, tgt_text in pairs:
            src_tok_seq = src_tokenizer(src_text)
            tgt_tok_seq = tgt_tokenizer(tgt_text)
            if max_length is not None and (len(src_tok_seq) > max_length or len(tgt_tok_seq) > max_length):
                continue
            src_tokens = [src_vocab.bos_idx] + src_vocab.lookup_tokens(src_tok_seq) + [src_vocab.eos_idx]
            tgt_tokens = [tgt_vocab.bos_idx] + tgt_vocab.lookup_tokens(tgt_tok_seq) + [tgt_vocab.eos_idx]
            self.examples.append((src_tokens, tgt_tokens))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.examples[idx]


def collate_fn(batch: Sequence[Tuple[List[int], List[int]]], src_pad_idx: int, tgt_pad_idx: int) -> Tuple[Tensor, Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_tensor = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in src_batch], padding_value=src_pad_idx)
    tgt_tensor = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in tgt_batch], padding_value=tgt_pad_idx)
    return src_tensor, tgt_tensor


class NoamScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int, factor: float = 1.0, step: int = 0) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = step
        self._update_learning_rate()

    def _compute_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step if self._step > 0 else 1
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        return self.factor * (self.d_model ** -0.5) * min(arg1, arg2)

    def _update_learning_rate(self) -> float:
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def step(self) -> float:
        self._step += 1
        return self._update_learning_rate()

    def state_dict(self) -> Dict[str, int]:
        return {"step": self._step}

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        self._step = state_dict.get("step", 0)
        self._update_learning_rate()


# ---------------------------
# 模型结构
# ---------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor],
        tgt_mask: Optional[Tensor],
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ) -> Tensor:
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.generator(output)


# ---------------------------
# 训练循环与评估
# ---------------------------


def generate_square_subsequent_mask(size: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(size, size, device=device, dtype=torch.bool),
        diagonal=1,
    )


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 10
    lr_factor: float = 1.0
    warmup_steps: int = 4000
    max_train_samples: Optional[int] = None
    max_valid_samples: Optional[int] = 2000
    min_freq: int = 2
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    clip_norm: float = 1.0
    grad_accum_steps: int = 2
    label_smoothing: float = 0.1
    sp_vocab_size: int = 16000
    sp_character_coverage: float = 1.0
    force_retrain_sp: bool = False
    max_seq_length: Optional[int] = 250


def create_dataloaders(
    data_dir: Path,
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, Vocab, Vocab, Path, Path]:
    train_pairs = read_parallel_corpus(
        data_dir / "train.tags.en-de.en",
        data_dir / "train.tags.en-de.de",
        max_samples=config.max_train_samples,
    )
    random.shuffle(train_pairs)
    valid_pairs = read_parallel_corpus(
        data_dir / "IWSLT17.TED.dev2010.en-de.en.xml",
        data_dir / "IWSLT17.TED.dev2010.en-de.de.xml",
        max_samples=config.max_valid_samples,
    )

    src_train_texts = [src for src, _ in train_pairs] + [src for src, _ in valid_pairs]
    tgt_train_texts = [tgt for _, tgt in train_pairs] + [tgt for _, tgt in valid_pairs]

    src_sp_model = data_dir / "spm_en.model"
    tgt_sp_model = data_dir / "spm_de.model"

    train_sentencepiece_model(
        src_train_texts,
        src_sp_model,
        vocab_size=config.sp_vocab_size,
        character_coverage=config.sp_character_coverage,
        retrain=config.force_retrain_sp,
    )
    train_sentencepiece_model(
        tgt_train_texts,
        tgt_sp_model,
        vocab_size=config.sp_vocab_size,
        character_coverage=config.sp_character_coverage,
        retrain=config.force_retrain_sp,
    )

    src_sp = spm.SentencePieceProcessor(model_file=str(src_sp_model))
    tgt_sp = spm.SentencePieceProcessor(model_file=str(tgt_sp_model))

    src_tokenizer = sentencepiece_tokenizer(src_sp)
    tgt_tokenizer = sentencepiece_tokenizer(tgt_sp)

    src_tokenized_train = [src_tokenizer(src) for src, _ in train_pairs]
    tgt_tokenized_train = [tgt_tokenizer(tgt) for _, tgt in train_pairs]
    src_tokenized_valid = [src_tokenizer(src) for src, _ in valid_pairs]
    tgt_tokenized_valid = [tgt_tokenizer(tgt) for _, tgt in valid_pairs]

    max_vocab = config.sp_vocab_size + len(SPECIAL_TOKENS)
    src_vocab = build_vocab(
        src_tokenized_train + src_tokenized_valid,
        max_size=max_vocab,
        min_freq=config.min_freq,
    )
    tgt_vocab = build_vocab(
        tgt_tokenized_train + tgt_tokenized_valid,
        max_size=max_vocab,
        min_freq=config.min_freq,
    )

    train_dataset = TranslationDataset(
        train_pairs,
        src_tokenizer,
        tgt_tokenizer,
        src_vocab,
        tgt_vocab,
        max_length=config.max_seq_length,
    )
    valid_dataset = TranslationDataset(
        valid_pairs,
        src_tokenizer,
        tgt_tokenizer,
        src_vocab,
        tgt_vocab,
        max_length=config.max_seq_length,
    )

    collate = lambda batch: collate_fn(batch, src_pad_idx=src_vocab.pad_idx, tgt_pad_idx=tgt_vocab.pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    return train_loader, valid_loader, src_vocab, tgt_vocab, src_sp_model, tgt_sp_model


def train_one_epoch(
    model: TransformerModel,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    src_pad_idx: int,
    tgt_pad_idx: int,
    clip_norm: float,
    scheduler: Optional[NoamScheduler] = None,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    step_in_epoch = 0
    for batch_idx, (src, tgt) in enumerate(dataloader, start=1):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        src_padding_mask = (src == src_pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt_input == tgt_pad_idx).transpose(0, 1)
        src_mask = None
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0), device=device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        raw_loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss = raw_loss / grad_accum_steps

        loss.backward()
        step_in_epoch += 1
        if step_in_epoch % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += raw_loss.item()

    if step_in_epoch % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: TransformerModel,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    src_pad_idx: int,
    tgt_pad_idx: int,
) -> float:
    model.eval()
    total_loss = 0.0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        src_padding_mask = (src == src_pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt_input == tgt_pad_idx).transpose(0, 1)
        src_mask = None
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0), device=device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(dataloader)


def fit(config: TrainingConfig, data_dir: Path, device: torch.device, resume_path: Optional[Path] = None) -> None:
    print(f"使用设备: {device}")

    train_loader, valid_loader, src_vocab, tgt_vocab, src_sp_model, tgt_sp_model = create_dataloaders(data_dir, config)

    model = TransformerModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, config.d_model, config.warmup_steps, factor=config.lr_factor)

    start_epoch = 1
    best_valid = float("inf")

    if resume_path is not None:
        print(f"从 {resume_path} 恢复训练")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        saved_config = checkpoint.get("config")
        if saved_config:
            for key, value in saved_config.items():
                if hasattr(config, key) and key not in {"num_epochs", "max_train_samples", "max_valid_samples"}:
                    setattr(config, key, value)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_valid = checkpoint.get("best_valid", float("inf"))

    end_epoch = start_epoch + config.num_epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
            src_vocab.pad_idx,
            tgt_vocab.pad_idx,
            config.clip_norm,
            scheduler=scheduler,
            grad_accum_steps=config.grad_accum_steps,
        )
        valid_loss = evaluate(
            model,
            criterion,
            valid_loader,
            device,
            src_vocab.pad_idx,
            tgt_vocab.pad_idx,
        )
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]
        ppl = math.exp(valid_loss) if valid_loss < 20 else float("inf")
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, ppl={ppl:.2f}, lr={lr:.6f}, time={elapsed:.1f}s")
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "src_vocab": src_vocab.itos,
                    "tgt_vocab": tgt_vocab.itos,
                    "config": config.__dict__,
                    "epoch": epoch,
                    "valid_loss": valid_loss,
                    "best_valid": best_valid,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "src_sp_model": src_sp_model.name,
                    "tgt_sp_model": tgt_sp_model.name,
                },
                data_dir / "transformer_en_de.pt",
            )
            print("保存更优模型到 transformer_en_de.pt")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer model on the IWSLT17 EN-DE dataset.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="选择训练设备。auto 在有 GPU 时使用 GPU。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--epochs", type=int, help="覆盖默认训练轮数。")
    parser.add_argument("--batch-size", type=int, help="覆盖默认 batch size。")
    parser.add_argument("--max-train-samples", type=int, help="限制训练样本数量。")
    parser.add_argument("--max-valid-samples", type=int, help="限制验证样本数量。")
    parser.add_argument("--sp-vocab-size", type=int, help="SentencePiece 词表大小。")
    parser.add_argument("--sp-character-coverage", type=float, help="SentencePiece character coverage。")
    parser.add_argument("--force-retrain-sp", action="store_true", help="无论文件是否存在都重新训练 SentencePiece 模型。")
    parser.add_argument("--warmup-steps", type=int, help="学习率 warmup 步数。")
    parser.add_argument("--lr-factor", type=float, help="Noam 调度器的缩放因子。")
    parser.add_argument("--grad-accum-steps", type=int, help="梯度累积步数。")
    parser.add_argument("--label-smoothing", type=float, help="标签平滑系数。")
    parser.add_argument("--max-seq-length", type=int, help="过滤超过该长度的样本。")
    parser.add_argument("--d-model", type=int, help="Transformer 模型维度。")
    parser.add_argument("--dim-feedforward", type=int, help="前馈网络隐藏层维度。")
    parser.add_argument("--num-encoder-layers", type=int, help="编码器层数。")
    parser.add_argument("--num-decoder-layers", type=int, help="解码器层数。")
    parser.add_argument("--nhead", type=int, help="注意力头数。")
    parser.add_argument("--resume", type=Path, help="从已有 checkpoint 继续训练。")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(__file__).parent

    config = TrainingConfig()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples
    if args.max_valid_samples is not None:
        config.max_valid_samples = args.max_valid_samples
    if args.sp_vocab_size is not None:
        config.sp_vocab_size = args.sp_vocab_size
    if args.sp_character_coverage is not None:
        config.sp_character_coverage = args.sp_character_coverage
    if args.force_retrain_sp:
        config.force_retrain_sp = True
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.lr_factor is not None:
        config.lr_factor = args.lr_factor
    if args.grad_accum_steps is not None:
        config.grad_accum_steps = args.grad_accum_steps
    if args.label_smoothing is not None:
        config.label_smoothing = args.label_smoothing
    if args.max_seq_length is not None:
        config.max_seq_length = args.max_seq_length
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.dim_feedforward is not None:
        config.dim_feedforward = args.dim_feedforward
    if args.num_encoder_layers is not None:
        config.num_encoder_layers = args.num_encoder_layers
    if args.num_decoder_layers is not None:
        config.num_decoder_layers = args.num_decoder_layers
    if args.nhead is not None:
        config.nhead = args.nhead

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("未检测到可用 GPU，自动改用 CPU。")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fit(config, data_dir, device, resume_path=args.resume)


if __name__ == "__main__":
    main()

