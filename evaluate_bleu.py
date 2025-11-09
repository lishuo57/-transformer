import argparse
import math
from pathlib import Path
from typing import List, Tuple

import sacrebleu
import sentencepiece as spm
import torch
import torch.nn.functional as F

from train_transformer import (
    TransformerModel,
    TrainingConfig,
    Vocab,
    generate_square_subsequent_mask,
    read_parallel_corpus,
)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[TransformerModel, Vocab, Vocab, TrainingConfig, spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config_dict = ckpt["config"]
    config = TrainingConfig(**config_dict)

    src_vocab = Vocab.from_itos(ckpt["src_vocab"])
    tgt_vocab = Vocab.from_itos(ckpt["tgt_vocab"])

    ckpt_dir = checkpoint_path.parent
    src_sp_model_name = ckpt.get("src_sp_model", "spm_en.model")
    tgt_sp_model_name = ckpt.get("tgt_sp_model", "spm_de.model")
    src_sp = spm.SentencePieceProcessor(model_file=str(ckpt_dir / src_sp_model_name))
    tgt_sp = spm.SentencePieceProcessor(model_file=str(ckpt_dir / tgt_sp_model_name))

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
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, src_vocab, tgt_vocab, config, src_sp, tgt_sp


def encode_sentence(sentence: str, vocab: Vocab, sp: spm.SentencePieceProcessor) -> List[int]:
    pieces = sp.encode(sentence.lower(), out_type=str)
    tokens = [vocab.bos_idx] + vocab.lookup_tokens(pieces) + [vocab.eos_idx]
    return tokens


def greedy_decode(
    model: TransformerModel,
    src_tokens: torch.Tensor,
    src_pad_idx: int,
    tgt_vocab: Vocab,
    max_len: int = 100,
) -> List[int]:
    device = src_tokens.device
    src_padding_mask = (src_tokens.squeeze(1) == src_pad_idx).unsqueeze(0)

    src_emb = model.pos_encoder(model.src_embed(src_tokens) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    ys = torch.full((1, 1), tgt_vocab.bos_idx, dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device=device)
        tgt_padding_mask = (ys.squeeze(1) == tgt_vocab.pad_idx).unsqueeze(0)
        tgt_emb = model.pos_decoder(model.tgt_embed(ys) * math.sqrt(model.d_model))
        out = model.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        logits = model.generator(out[-1])
        next_token = logits.argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device, dtype=torch.long)], dim=0)
        if next_token == tgt_vocab.eos_idx:
            break
    return ys.squeeze(1).tolist()[1:]  # remove BOS


def apply_length_penalty(score: float, length: int, alpha: float) -> float:
    if alpha == 0.0:
        return score
    return score / (((5 + length) / 6) ** alpha)


def beam_search_decode(
    model: TransformerModel,
    src_tokens: torch.Tensor,
    src_pad_idx: int,
    tgt_vocab: Vocab,
    max_len: int,
    beam_size: int,
    length_penalty: float,
) -> List[int]:
    device = src_tokens.device
    src_padding_mask = (src_tokens.squeeze(1) == src_pad_idx).unsqueeze(0)
    src_emb = model.pos_encoder(model.src_embed(src_tokens) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    active_sequences: List[List[int]] = [[tgt_vocab.bos_idx]]
    active_scores = torch.zeros(1, device=device)
    finished: List[Tuple[List[int], float]] = []

    for _ in range(max_len):
        if not active_sequences:
            break

        seq_tensor = torch.tensor(active_sequences, dtype=torch.long, device=device).transpose(0, 1)
        tgt_mask = generate_square_subsequent_mask(seq_tensor.size(0), device=device)
        tgt_padding_mask = (seq_tensor == tgt_vocab.pad_idx).transpose(0, 1)
        tgt_emb = model.pos_decoder(model.tgt_embed(seq_tensor) * math.sqrt(model.d_model))

        expanded_memory = memory.expand(-1, len(active_sequences), -1)
        expanded_src_padding = src_padding_mask.expand(len(active_sequences), -1)

        decoder_out = model.transformer.decoder(
            tgt_emb,
            expanded_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=expanded_src_padding,
        )
        logits = model.generator(decoder_out[-1])
        log_probs = F.log_softmax(logits, dim=-1)

        topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)
        total_scores = active_scores.unsqueeze(1) + topk_log_probs
        total_scores = total_scores.view(-1)
        top_scores, top_positions = total_scores.topk(min(beam_size, total_scores.size(0)))

        new_sequences: List[List[int]] = []
        new_scores: List[float] = []

        for score, pos in zip(top_scores.tolist(), top_positions.tolist()):
            beam_idx = pos // beam_size
            token_rank = pos % beam_size
            token_id = topk_indices[beam_idx, token_rank].item()

            if token_id == tgt_vocab.pad_idx:
                continue

            candidate = active_sequences[beam_idx] + [token_id]
            if token_id == tgt_vocab.eos_idx:
                finished.append((candidate, score))
            else:
                new_sequences.append(candidate)
                new_scores.append(score)

        if not new_sequences:
            break

        keep = min(beam_size, len(new_sequences))
        new_scores_tensor = torch.tensor(new_scores, device=device)
        top_new_scores, indices = new_scores_tensor.topk(keep)
        active_sequences = [new_sequences[i] for i in indices.tolist()]
        active_scores = top_new_scores

    if finished:
        scored = [(seq, apply_length_penalty(score, len(seq) - 1, length_penalty)) for seq, score in finished]
        best_seq = max(scored, key=lambda x: x[1])[0]
    else:
        best_idx = int(torch.argmax(active_scores)) if active_scores.numel() else 0
        best_seq = active_sequences[best_idx]

    return best_seq[1:]


def ids_to_sentence(token_ids: List[int], vocab: Vocab, sp: spm.SentencePieceProcessor) -> str:
    pieces: List[str] = []
    for idx in token_ids:
        if idx == vocab.eos_idx:
            break
        if idx in (vocab.pad_idx,):
            continue
        token = vocab.itos[idx]
        if token == "<bos>":
            continue
        pieces.append(token)
    if not pieces:
        return ""
    return sp.decode(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Transformer model using sacreBLEU.")
    parser.add_argument("--model", type=Path, default=Path("transformer_en_de.pt"), help="模型权重路径")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="数据目录（包含 IWSLT 文件）")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="选择评估的数据集")
    parser.add_argument("--output", type=Path, default=Path("preds.txt"), help="保存译文的文件路径")
    parser.add_argument("--refs", type=Path, default=Path("refs.txt"), help="保存参考译文的文件路径")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="运行设备")
    parser.add_argument("--max-len", type=int, default=100, help="译文生成的最大长度")
    parser.add_argument("--beam-size", type=int, default=5, help="束搜索宽度，设置为 1 等价于贪心解码。")
    parser.add_argument("--length-penalty", type=float, default=0.6, help="束搜索长度惩罚系数。")
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("未检测到 GPU，改用 CPU。")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, src_vocab, tgt_vocab, _, src_sp, tgt_sp = load_model(args.model, device)

    if args.split == "dev":
        src_file = args.data_dir / "IWSLT17.TED.dev2010.en-de.en.xml"
        tgt_file = args.data_dir / "IWSLT17.TED.dev2010.en-de.de.xml"
    else:
        src_file = args.data_dir / "IWSLT17.TED.tst2010.en-de.en.xml"
        tgt_file = args.data_dir / "IWSLT17.TED.tst2010.en-de.de.xml"

    pairs = read_parallel_corpus(src_file, tgt_file, max_samples=None)
    references = [tgt for _, tgt in pairs]

    predictions: List[str] = []
    beam_size = max(1, args.beam_size)
    length_penalty = max(0.0, args.length_penalty)

    with torch.no_grad():
        for idx, (src_text, _) in enumerate(pairs, start=1):
            src_ids = encode_sentence(src_text, src_vocab, src_sp)
            src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)
            if beam_size == 1:
                generated_ids = greedy_decode(model, src_tensor, src_vocab.pad_idx, tgt_vocab, max_len=args.max_len)
            else:
                generated_ids = beam_search_decode(
                    model,
                    src_tensor,
                    src_vocab.pad_idx,
                    tgt_vocab,
                    max_len=args.max_len,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )
            prediction = ids_to_sentence(generated_ids, tgt_vocab, tgt_sp)
            predictions.append(prediction)
            if idx % 50 == 0:
                print(f"已生成 {idx}/{len(pairs)} 条译文")

    args.output.write_text("\n".join(predictions), encoding="utf-8")
    args.refs.write_text("\n".join(references), encoding="utf-8")

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"sacreBLEU = {bleu.score:.2f}")
    print(f"详情: {bleu}")


if __name__ == "__main__":
    main()

