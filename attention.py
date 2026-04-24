"""Multi-head self-attention mechanism + SVD word embeddings (numpy only).

Embeddings are learned from word co-occurrence: build a PPMI
(positive pointwise mutual information) matrix on a small curated
corpus, then take top singular vectors. This is the classic
word-embedding recipe predating word2vec — no gradient training
required, and the embeddings capture real distributional semantics.

The attention projections (W_q, W_k, W_v, W_o) are seeded random. The
forward pass mirrors the paper ("Attention Is All You Need"): scaled
dot-product multi-head self-attention, with sinusoidal positional
encoding added to embeddings. No layer norm, no FFN — just the
attention block, to keep the visualization focused.

Run directly to produce attention_model.js for the browser viz:
    python attention.py
"""

import json

import numpy as np

EMBED_DIM = 32
NUM_HEADS = 4
MAX_SEQ = 24
WINDOW = 4
SEED = 42

CORPUS = [
    "the cat sat on the mat .",
    "the cat chased the mouse .",
    "the dog chased the cat up the tree .",
    "the dog barked at the stranger .",
    "a small cat slept on the couch .",
    "the big dog ran in the park .",
    "birds fly high in the sky .",
    "fish swim deep in the ocean .",
    "the fish swam under the boat .",
    "a bird landed on the fence .",
    "she went to the store .",
    "he walked to the park .",
    "she opened the door slowly .",
    "he closed the window quickly .",
    "i saw a bird fly away .",
    "i heard a dog bark loudly .",
    "the boy read a book .",
    "the girl played with her dog .",
    "the boy threw the ball .",
    "the girl caught the ball .",
    "we had dinner at the restaurant .",
    "we ate pizza for lunch .",
    "they went home after the movie .",
    "they watched a movie together .",
    "the teacher asked a question .",
    "the student answered the question .",
    "the doctor helped the patient .",
    "the patient thanked the doctor .",
    "the king ruled the kingdom .",
    "the queen sat on the throne .",
    "the child hugged the mother .",
    "the mother kissed the child .",
    "the father drove the car .",
    "the car stopped at the light .",
    "the river runs through the valley .",
    "the mountain stood tall and silent .",
    "the sun rose over the hill .",
    "the moon shone on the lake .",
    "stars twinkled in the night sky .",
    "rain fell on the roof .",
    "snow covered the ground .",
    "wind blew through the trees .",
    "leaves fell from the tree .",
    "flowers bloomed in the garden .",
    "the garden was full of flowers .",
    "she planted a tree in the garden .",
    "he cut the grass in the yard .",
    "the man wrote a letter .",
    "the woman read the letter .",
    "the friend called his friend .",
    "music played in the room .",
    "the singer sang a song .",
    "the painter painted a picture .",
    "the writer wrote a story .",
    "the baker baked the bread .",
    "the cook cooked the meal .",
    "the farmer grew the crops .",
    "the soldier guarded the gate .",
    "the sailor sailed the ship .",
    "the ship sailed across the sea .",
    "the train moved along the track .",
    "the plane flew above the clouds .",
    "the bus stopped at the corner .",
    "the bicycle rolled down the hill .",
    "the horse ran across the field .",
    "the field was green and open .",
    "the forest was dark and quiet .",
    "a wolf howled in the forest .",
    "a deer jumped over the fence .",
    "a rabbit hopped through the grass .",
    "the child laughed at the joke .",
    "the old man smiled at the child .",
    "the baby cried in the night .",
    "the nurse held the baby .",
    "the team won the game .",
    "the player scored a goal .",
    "the crowd cheered for the team .",
    "the coach trained the players .",
    "the chef prepared a meal .",
    "the waiter brought the food .",
    "the food tasted good .",
    "the coffee was hot and strong .",
    "the water was cold and clear .",
    "the ice melted in the sun .",
    "the fire burned through the night .",
    "the candle flickered on the table .",
    "the lamp lit the room .",
    "the room was warm and cozy .",
    "the house stood on the hill .",
    "a bridge crossed over the river .",
    "the road led to the city .",
    "the city was busy and loud .",
    "the village was quiet and small .",
    "she wore a blue dress .",
    "he wore a red shirt .",
    "the sky turned orange at sunset .",
    "the leaves turned yellow in autumn .",
    "she smiled at him gently .",
    "he waved at her from the window .",
    "the clock ticked on the wall .",
    "time passed slowly in the room .",
]


def build_vocab(corpus):
    words = set()
    for s in corpus:
        words.update(s.split())
    vocab = ["<pad>", "<unk>"] + sorted(words)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    return vocab, word_to_id


def cooccurrence_matrix(corpus, word_to_id, window=WINDOW):
    V = len(word_to_id)
    C = np.zeros((V, V), dtype=np.float32)
    for s in corpus:
        ids = [word_to_id[w] for w in s.split()]
        for i, ci in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j == i:
                    continue
                C[ci, ids[j]] += 1.0 / abs(j - i)
    return C


def ppmi(C):
    total = C.sum()
    row_sums = C.sum(axis=1, keepdims=True) + 1e-9
    col_sums = C.sum(axis=0, keepdims=True) + 1e-9
    expected = (row_sums * col_sums) / (total + 1e-9)
    pmi = np.log((C + 1e-9) / (expected + 1e-9))
    return np.maximum(pmi, 0.0)


def svd_embeddings(M, d):
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    E = U[:, :d] * np.sqrt(S[:d])
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return (E / norms).astype(np.float32)


def positional_encoding(seq_len, d):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d)
    pe = np.zeros((seq_len, d), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def init_attention_params(d, seed=SEED):
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(d)
    return {
        "W_q": rng.normal(0, scale, (d, d)).astype(np.float32),
        "W_k": rng.normal(0, scale, (d, d)).astype(np.float32),
        "W_v": rng.normal(0, scale, (d, d)).astype(np.float32),
        "W_o": rng.normal(0, scale, (d, d)).astype(np.float32),
    }


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def multi_head_attention(x, params, num_heads):
    T, D = x.shape
    d_h = D // num_heads
    q = x @ params["W_q"]
    k = x @ params["W_k"]
    v = x @ params["W_v"]
    q_h = q.reshape(T, num_heads, d_h).transpose(1, 0, 2)
    k_h = k.reshape(T, num_heads, d_h).transpose(1, 0, 2)
    v_h = v.reshape(T, num_heads, d_h).transpose(1, 0, 2)
    scores = q_h @ k_h.transpose(0, 2, 1) / np.sqrt(d_h)
    attn = softmax(scores, axis=-1)
    out_h = attn @ v_h
    out = out_h.transpose(1, 0, 2).reshape(T, D)
    out = out @ params["W_o"]
    return out, attn


def forward(token_ids, E, params, PE, num_heads):
    x = E[token_ids] + PE[: len(token_ids)]
    out, attn = multi_head_attention(x, params, num_heads)
    return x, out, attn


def build_model():
    vocab, word_to_id = build_vocab(CORPUS)
    C = cooccurrence_matrix(CORPUS, word_to_id, window=WINDOW)
    M = ppmi(C)
    E = svd_embeddings(M, EMBED_DIM)
    params = init_attention_params(EMBED_DIM, seed=SEED)
    PE = positional_encoding(MAX_SEQ, EMBED_DIM)
    return vocab, word_to_id, E, params, PE


def export_js(out_path="attention_model.js"):
    vocab, word_to_id, E, params, PE = build_model()

    def arr(a):
        return {"shape": list(a.shape), "data": np.round(a, 5).flatten().tolist()}

    payload = {
        "config": {
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "max_seq": MAX_SEQ,
        },
        "vocab": vocab,
        "E": arr(E),
        "PE": arr(PE),
        "W_q": arr(params["W_q"]),
        "W_k": arr(params["W_k"]),
        "W_v": arr(params["W_v"]),
        "W_o": arr(params["W_o"]),
        "sample_sentences": [
            "the cat sat on the mat",
            "the dog chased the cat up the tree",
            "birds fly high in the sky",
            "she went to the store",
            "the boy read a book",
            "the sun rose over the hill",
        ],
    }

    with open(out_path, "w") as f:
        f.write("window.ATTENTION_MODEL = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")

    print(f"Wrote {out_path}")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Embed dim:  {EMBED_DIM}  |  Heads: {NUM_HEADS}  |  Head dim: {EMBED_DIM // NUM_HEADS}")
    print(f"  Corpus:     {len(CORPUS)} sentences")


if __name__ == "__main__":
    export_js()
