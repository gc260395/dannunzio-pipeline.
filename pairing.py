"""
Referring-expression pairing for the D'Annunzio parallel corpus.

Reads two TEI files annotated with the dannunzio-coref customization and their
stand-off sentence alignment, and writes one row per correspondence: either a
pair of mentions or a single unpaired mention.

Usage:
    python pairing.py --source 1894-dannunzio-01.xml \\
                      --target 1913-ishikawa-01.xml \\
                      --alignment 1894_1913-alignments-01.xml \\
                      --out data/1894-1913-mentions.csv

Note: read the output with pd.read_csv(path, keep_default_na=False, na_values=[]).
"""

import argparse
import re
from collections import defaultdict, namedtuple
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

EMBEDDER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDER_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"

W_SEM = 0.875
W_POS = 0.125

ZERO_POS = "NULL"
NOM_CASE = "nsubj"
OBJ_CASE = "obj"
REFLEX_POS = "PRON-REFLEX"
NOM_POS = {"PRON", "PROPN", "NOUN"}
EXPL_POS = NOM_POS | {"DET"}

_STATS = defaultdict(int)

Span = namedtuple(
    "Span",
    ["sentence_id", "sentence_index", "ref", "text", "pos", "case",
     "idx_in_sent", "sent_text", "char_start", "char_end", "zero_sem_text"],
)


# ------- Similarity ------- 

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        kwargs = {"revision": EMBEDDER_REVISION} if EMBEDDER_REVISION else {}
        _embedder = SentenceTransformer(EMBEDDER_ID, **kwargs)
    return _embedder


@lru_cache(maxsize=20000)
def _embed(text):
    text = (text or "").strip()
    if not text:
        return None
    return _get_embedder().encode(text, normalize_embeddings=True)


def semantic_sim(a, b):
    va, vb = _embed(a), _embed(b)
    if va is None or vb is None:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(va, vb))))


# ------- TEI parsing ------- 

def canon_ref(x):
    if not x:
        return "MISSING"
    parts = [p.lstrip("#") for p in re.split(r"\s+", str(x).strip()) if p]
    return " ".join(parts) if parts else "MISSING"


def is_explicit(s):
    return s.pos in EXPL_POS


def is_explicit_nom(s):
    # DET is excluded here: an Italian possessive determiner is annotated nmod
    # and can therefore never anchor a zero.
    return s.case == NOM_CASE and s.pos in NOM_POS


def is_zero_nom(s):
    return s.pos == ZERO_POS and s.case == NOM_CASE


def parse_tei(tei_path, zero_context_mode):
    """zero_context_mode: "right" for Italian, "left" for Japanese.

    The span used for the semantic comparison runs from the predicate to the
    next zero predicate in Italian, and from the previous zero predicate to the
    current one in Japanese, following the position of the predicate's arguments
    in each language.
    """
    soup = BeautifulSoup(Path(tei_path).read_text(encoding="utf-8"), "lxml-xml")

    def is_zero(spec):
        return spec["pos"] == ZERO_POS and spec["case"] == NOM_CASE

    def offsets(txt, specs):
        out, cursor = [], 0
        for spec in specs:
            surface = spec["text"] or ""
            start = txt.find(surface, cursor) if surface else -1
            if surface and start < 0:
                start = txt.find(surface)
                if 0 <= start < cursor:
                    _STATS["offset_backtrack"] += 1
            end = start + len(surface) if start >= 0 else -1
            if end >= 0:
                cursor = end
            spec = dict(spec)
            spec["char_start"], spec["char_end"] = start, end
            out.append(spec)
        return out

    def zero_spans(txt, specs, mode):
        idx = [i for i, sp in enumerate(specs)
               if is_zero(sp) and sp["char_start"] >= 0 and sp["char_end"] >= 0]
        for n, i in enumerate(idx):
            cur = specs[i]
            if mode == "right":
                start = cur["char_start"]
                end = specs[idx[n + 1]]["char_start"] if n + 1 < len(idx) else len(txt)
            else:
                start = specs[idx[n - 1]]["char_end"] if n > 0 else 0
                end = cur["char_end"]
            local = txt[start:end].strip()
            if not local:
                local = cur["text"]
                _STATS["zero_span_empty"] += 1
            specs[i]["zero_sem_text"] = local
        for sp in specs:
            sp.setdefault("zero_sem_text", "")
        return specs

    sent_text, spans_by_sid = {}, defaultdict(list)
    for sent_idx, s in enumerate(soup.find_all("s")):
        sid = s.get("xml:id")
        txt = s.get_text()
        sent_text[sid] = txt
        specs = [{
            "sentence_id": sid,
            "sentence_index": sent_idx,
            "ref": canon_ref(rs.get("ref")),
            "text": rs.get_text(),
            "pos": rs.get("pos"),
            "case": rs.get("case"),
            "idx_in_sent": k,
            "sent_text": txt,
        } for k, rs in enumerate(s.find_all("rs"))]
        specs = zero_spans(txt, offsets(txt, specs), zero_context_mode)
        for sp in specs:
            spans_by_sid[sid].append(Span(**{f: sp[f] for f in Span._fields}))
    return sent_text, spans_by_sid


def parse_alignments(link_path, src_path, tgt_path):
    src_key, tgt_key = Path(src_path).name, Path(tgt_path).name
    soup = BeautifulSoup(Path(link_path).read_text(encoding="utf-8"), "lxml-xml")
    blocks, skipped = [], 0
    for link in soup.find_all("link"):
        ids = link.get("target", "").split()
        src_ids = [x.split("#")[-1] for x in ids if src_key in x]
        tgt_ids = [x.split("#")[-1] for x in ids if tgt_key in x]
        if not src_ids or not tgt_ids:
            skipped += 1
            continue
        if len(src_ids) > 1 and len(tgt_ids) == 1:
            atype = "many-to-one"
        elif len(src_ids) == 1 and len(tgt_ids) > 1:
            atype = "one-to-many"
        else:
            atype = "one-to-one"
        blocks.append((src_ids, tgt_ids, atype))
    if not blocks:
        raise ValueError(f"no alignment blocks matched {src_key} / {tgt_key}")
    if skipped:
        print(f"  {skipped} one-sided link(s) skipped")
    return blocks


# ------- cost -------

def zero_text(span):
    local = (span.zero_sem_text or "").strip()
    return local if local else (span.text or "").strip()


def zero_cost(z_src, z_tgt, rel_src, rel_tgt):
    sim = semantic_sim(zero_text(z_src), zero_text(z_tgt))
    return W_SEM * (1.0 - sim) + W_POS * abs(rel_src - rel_tgt)


def block_offsets(ids, sent_text):
    off, cum = {}, 0
    for sid in ids:
        off[sid] = cum
        cum += len(sent_text.get(sid, ""))
    return off, max(cum, 1)


def rel_pos(span, offsets, total):
    if span.char_start is None or span.char_start < 0:
        return 0.5
    return (offsets.get(span.sentence_id, 0) + span.char_start) / total


# ------- pairing -------

def pair_entity(src_spans, tgt_spans, block_pos, src_off, tgt_off):
    """Every step consumes on both sides, so each mention ends up in exactly
    one pair or one leftover."""

    def greedy(A, B, keys, tag):
        matched, used, a_left = [], set(), []
        for a in A:
            hit = -1
            for j, b in enumerate(B):
                if j in used:
                    continue
                if all(getattr(a, k) == getattr(b, k) for k in keys):
                    matched.append((a, b, tag, None))
                    used.add(j)
                    hit = j
                    break
            if hit < 0:
                a_left.append(a)
        return matched, a_left, [b for j, b in enumerate(B) if j not in used]

    def zero_to_zero(S, T):
        m, n = len(S), len(T)
        if m == 0 or n == 0:
            return [], S, T
        if m == 1 and n == 1:
            return [(S[0], T[0], "zero_zero_single", None)], [], []
        C = np.zeros((m, n))
        for i, s in enumerate(S):
            for j, t in enumerate(T):
                C[i, j] = zero_cost(s, t, rel_pos(s, *src_off), rel_pos(t, *tgt_off))
        rows, ui, uj = [], set(), set()
        for i, j in zip(*linear_sum_assignment(C)):
            rows.append((S[i], T[j], "zero_zero_semantic", float(C[i, j])))
            ui.add(i)
            uj.add(j)
        return (rows,
                [S[i] for i in range(m) if i not in ui],
                [T[j] for j in range(n) if j not in uj])

    def order(x):
        return (block_pos.get(x.sentence_id, 10 ** 9), x.idx_in_sent)

    def zero_to_anchor(zeros, anchors, tag):
        by_ref = defaultdict(list)
        for e in anchors:
            if is_explicit_nom(e):
                by_ref[e.ref].append(e)
        for r in by_ref:
            by_ref[r].sort(key=order)
        rows, used, left = [], set(), []
        for z in sorted(zeros, key=order):
            cands = [e for e in by_ref.get(z.ref, []) if e not in used]
            if not cands:
                left.append(z)
                continue
            e = cands[0]
            used.add(e)
            rows.append((z, e, tag, None) if tag == "zero_to_tgt_anchor"
                        else (e, z, tag, None))
        return rows, left

    S_all, T_all = list(src_spans), list(tgt_spans)
    S_exp = [s for s in S_all if is_explicit(s)]
    T_exp = [t for t in T_all if is_explicit(t)]
    S_zero = [s for s in S_all if is_zero_nom(s)]
    T_zero = [t for t in T_all if is_zero_nom(t)]

    rows = []
    for keys, tag in [(("ref", "pos", "case"), "explicit_full"),
                      (("ref", "case"), "explicit_ref_case"),
                      (("ref", "pos"), "explicit_ref_pos")]:
        m, S_exp, T_exp = greedy(S_exp, T_exp, keys, tag)
        rows.extend(m)
    m, S_exp_left, T_exp_left = greedy(
        [s for s in S_exp if s.pos != REFLEX_POS],
        [t for t in T_exp if t.pos != REFLEX_POS],
        ("ref",), "explicit_ref_only")
    rows.extend(m)

    zz, S_zero, T_zero = zero_to_zero(S_zero, T_zero)
    rows.extend(zz)

    if S_zero and T_exp_left:
        r, S_zero = zero_to_anchor(S_zero, T_exp_left, "zero_to_tgt_anchor")
        rows.extend(r)
        consumed = {b for (_, b, _, _) in r}
        T_exp_left = [t for t in T_exp_left if t not in consumed]
    if T_zero and S_exp_left:
        r, T_zero = zero_to_anchor(T_zero, S_exp_left, "zero_to_src_anchor")
        rows.extend(r)
        consumed = {a for (a, _, _, _) in r}
        S_exp_left = [s for s in S_exp_left if s not in consumed]

    used_tgt = {b for (_, b, _, _) in rows}
    m, _, _ = greedy(
        [s for s in S_all if s.pos == REFLEX_POS and s.case == OBJ_CASE],
        [t for t in T_all if t.case == OBJ_CASE and t.pos in NOM_POS
         and t not in used_tgt],
        ("ref",), "reflex_obj")
    rows.extend(m)

    used_src = {a for (a, _, _, _) in rows}
    used_tgt = {b for (_, b, _, _) in rows}
    return (rows,
            [s for s in S_all if s not in used_src],
            [t for t in T_all if t not in used_tgt])


# ------- output ------- 

def _row(pair_num, src_ids, tgt_ids, atype, src_sents, tgt_sents,
         a=None, b=None, pair_type="", zero_pair_cost=""):
    return {
        "sentence_pair": pair_num,
        "alignment_type": atype,
        "source_sentence_ids": " ".join(src_ids),
        "target_sentence_ids": " ".join(tgt_ids),
        "source_sents": src_sents,
        "target_sents": tgt_sents,
        "source_text": a.text if a else "MISSING",
        "target_text": b.text if b else "MISSING",
        "source_ref": a.ref if a else "MISSING",
        "target_ref": b.ref if b else "MISSING",
        "source_pos": a.pos if a else "MISSING",
        "target_pos": b.pos if b else "MISSING",
        "source_case": a.case if a else "MISSING",
        "target_case": b.case if b else "MISSING",
        "pair_type": pair_type,
        "zero_pair_cost": zero_pair_cost,
    }


def build_dataset(alignment_file, source_xml, target_xml):
    src_sent_text, src_spans = parse_tei(source_xml, "right")
    tgt_sent_text, tgt_spans = parse_tei(target_xml, "left")
    blocks = parse_alignments(alignment_file, source_xml, target_xml)

    out = []
    for pair_num, (src_ids, tgt_ids, atype) in enumerate(blocks, 1):
        S_block = [sp for sid in src_ids for sp in src_spans.get(sid, [])]
        T_block = [sp for sid in tgt_ids for sp in tgt_spans.get(sid, [])]
        block_pos = {sid: i for i, sid in enumerate(src_ids + tgt_ids)}
        src_off = block_offsets(src_ids, src_sent_text)
        tgt_off = block_offsets(tgt_ids, tgt_sent_text)
        src_sents = " ".join(src_sent_text[i] for i in src_ids if i in src_sent_text)
        tgt_sents = " ".join(tgt_sent_text[i] for i in tgt_ids if i in tgt_sent_text)
        common = (pair_num, src_ids, tgt_ids, atype, src_sents, tgt_sents)

        for ref in sorted({s.ref for s in S_block} | {t.ref for t in T_block}):
            rows, left_s, left_t = pair_entity(
                [s for s in S_block if s.ref == ref],
                [t for t in T_block if t.ref == ref],
                block_pos, src_off, tgt_off)
            for a, b, tag, cost in rows:
                out.append(_row(*common, a=a, b=b, pair_type=tag,
                                zero_pair_cost="" if cost is None else cost))
            for a in left_s:
                out.append(_row(*common, a=a, pair_type="leftover_source"))
            for b in left_t:
                out.append(_row(*common, b=b, pair_type="leftover_target"))

    return pd.DataFrame(out), blocks, src_spans, tgt_spans


def verify(df, blocks, src_spans, tgt_spans):
    """Coverage and conservation. Every mention inside an aligned block must
    appear exactly once, as one half of a pair or as a leftover."""
    for label, spans, side in [("source", src_spans, 0), ("target", tgt_spans, 1)]:
        total = sum(len(v) for v in spans.values())
        linked = {sid for b in blocks for sid in b[side]}
        covered = sum(len(v) for k, v in spans.items() if k in linked)
        print(f"  {label} mentions in aligned blocks: "
              f"{covered}/{total} ({covered / max(total, 1):.1%})")

    n_src = sum(len(src_spans.get(sid, [])) for b in blocks for sid in b[0])
    n_tgt = sum(len(tgt_spans.get(sid, [])) for b in blocks for sid in b[1])
    got_src = int((df.source_text != "MISSING").sum())
    got_tgt = int((df.target_text != "MISSING").sum())
    print(f"  conservation: source {got_src}/{n_src}  target {got_tgt}/{n_tgt}")
    if (got_src, got_tgt) != (n_src, n_tgt):
        raise AssertionError("mention dropped or double-counted")
    if _STATS:
        print(f"  stats: {dict(_STATS)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--alignment", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    df, blocks, src_spans, tgt_spans = build_dataset(
        args.alignment, args.source, args.target)
    if not args.no_verify:
        verify(df, blocks, src_spans, tgt_spans)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"  {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()