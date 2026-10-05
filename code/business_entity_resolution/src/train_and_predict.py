import os
import sys
import time
import re
import gc
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

import numpy as np
from rapidfuzz import fuzz
import anyascii
import lightgbm as lgb
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEGAL_SET = {
    "pvt", "ltd", "inc", "corp", "llc", "llp", "co", "plc", "gmbh",
    "sarl", "sasu", "sas", "eurl", "sci", "dba", "company", "corporation",
    "limited", "private", "incorporated"
}

RE_NUMS = re.compile(r"\d+")
RE_PUNCT = re.compile(r"[^\w\s]")

def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = anyascii.anyascii(text).lower()
    t = t.replace("&", " and ")
    t = RE_PUNCT.sub(" ", t)
    return " ".join(t.split())

def strip_legal(norm_text: str) -> str:
    tokens = [w for w in norm_text.split() if w not in LEGAL_SET]
    return " ".join(tokens) if tokens else norm_text

def extract_digits(text: str) -> List[str]:
    if not text:
        return []
    raw = RE_NUMS.findall(text)
    clean = {r.lstrip("0") for r in raw if len(r.lstrip("0")) >= 2}
    return sorted(clean, key=len, reverse=True)

class FastBlockingEngine:
    def __init__(self, max_cands_per_s1: int = 25):
        self.max_cands = max_cands_per_s1
        self.index = defaultdict(list)

    def _get_keys(self, name_norm: str, addr_norm: str, digits: List[str]) -> Set[str]:
        keys = set()
        name_no_leg = strip_legal(name_norm)
        tokens = name_no_leg.split()
        if tokens:
            keys.add("n1_" + tokens[0])
            if len(tokens) > 1:
                keys.add("n2_" + tokens[0] + "_" + tokens[1])
        for d in digits[:2]:
            if tokens:
                keys.add("nd_" + tokens[0] + "_" + d)
            if len(d) >= 3:
                keys.add("d_" + d)
        return keys

    def fit(self, targets: Dict[str, Tuple[str, str, List[str]]]):
        for tid, (tn, ta, t_dig) in targets.items():
            keys = self._get_keys(tn, ta, t_dig)
            for k in keys:
                if len(self.index[k]) < 50:
                    self.index[k].append(tid)

    def query(self, s1_name: str, s1_addr: str, s1_dig: List[str]) -> List[str]:
        keys = self._get_keys(s1_name, s1_addr, s1_dig)
        hits = Counter()
        for k in keys:
            for tid in self.index.get(k, []):
                hits[tid] += 1
        return [tid for tid, _ in hits.most_common(self.max_cands)]

def extract_features(sn: str, sa: str, s_dig: List[str], tn: str, ta: str, t_dig: List[str]) -> List[float]:
    n_set = fuzz.token_set_ratio(sn, tn) / 100.0
    n_sort = fuzz.token_sort_ratio(sn, tn) / 100.0
    n_ratio = fuzz.ratio(sn, tn) / 100.0
    a_set = fuzz.token_set_ratio(sa, ta) / 100.0 if sa and ta else 0.0
    a_ratio = fuzz.ratio(sa, ta) / 100.0 if sa and ta else 0.0
    
    s_set = set(s_dig)
    t_set = set(t_dig)
    overlap = len(s_set & t_set)
    has_digit_match = 1.0 if overlap > 0 else 0.0
    
    len_diff_n = abs(len(sn) - len(tn))
    len_diff_a = abs(len(sa) - len(ta))
    
    return [n_set, n_sort, n_ratio, a_set, a_ratio, has_digit_match, float(overlap), float(len_diff_n), float(len_diff_a)]

print("=" * 80)
print("PHASE 1: PREPARING TRAINING DATA")
print("=" * 80)

gt_map = defaultdict(set)
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for idx, line in enumerate(f):
        if idx >= 40000: break
        p = line.rstrip("\r\n").split("\t")
        sid = p[0].strip()
        m = [x.strip() for x in p[1].split(",") if x.strip()] if len(p) > 1 and p[1].strip() else []
        gt_map[sid] = set(m)

train_s1 = {}
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        sid = p[0].strip()
        if sid in gt_map:
            train_s1[sid] = (normalize_text(p[1]), normalize_text(p[2]), extract_digits(p[2]))

train_targets = {}
target_needed = set()
for mset in gt_map.values():
    target_needed.update(mset)

for fname in ["train_source2.tsv", "train_source3.tsv"]:
    fpath = os.path.join(TRAIN_DIR, fname)
    if not os.path.exists(fpath): continue
    with open(fpath, "r", encoding="utf-8") as f:
        f.readline()
        cnt = 0
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            tid = p[0].strip()
            if tid in target_needed or cnt < 30000:
                train_targets[tid] = (normalize_text(p[1]), normalize_text(p[2]), extract_digits(p[2]))
                cnt += 1

print(f"Fitting training blocker on {len(train_targets)} targets...")
blocker = FastBlockingEngine(max_cands_per_s1=20)
blocker.fit(train_targets)

X = []
y = []
for sid, (sn, sa, s_dig) in train_s1.items():
    true_set = gt_map[sid]
    for tid in true_set:
        if tid in train_targets:
            tn, ta, t_dig = train_targets[tid]
            feats = extract_features(sn, sa, s_dig, tn, ta, t_dig)
            X.append(feats)
            y.append(1)
            
    cands = blocker.query(sn, sa, s_dig)
    neg_count = 0
    for cid in cands:
        if cid not in true_set and cid in train_targets and neg_count < 3:
            tn, ta, t_dig = train_targets[cid]
            feats = extract_features(sn, sa, s_dig, tn, ta, t_dig)
            X.append(feats)
            y.append(0)
            neg_count += 1

X = np.array(X)
y = np.array(y)
print(f"Generated {len(X)} training vectors (Positives: {np.sum(y)}, Negatives: {len(y) - np.sum(y)})")

print("\n" + "=" * 80)
print("PHASE 2: TRAINING LIGHTGBM CLASSIFIER")
print("=" * 80)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.08,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[val_data]
)

print("\n" + "=" * 80)
print("PHASE 3: RUNNING TEST INFERENCE")
print("=" * 80)

s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
s1_order = []
s1_by_country = defaultdict(dict)

with open(s1_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        sid = p[0].strip()
        name = p[1].strip() if len(p) > 1 else ""
        addr = p[2].strip() if len(p) > 2 else ""
        country = p[3].strip() if len(p) > 3 else "UNKNOWN"
        s1_order.append(sid)
        s1_by_country[country][sid] = (normalize_text(name), normalize_text(addr), extract_digits(addr))

all_candidates = {}
all_scores = defaultdict(list)

for country, s1_dict in s1_by_country.items():
    print(f"\nEvaluating Country: {country} ({len(s1_dict)} S1 entities)...", flush=True)
    country_targets = {}
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        fpath = os.path.join(TEST_DIR, fname)
        if not os.path.exists(fpath): continue
        with open(fpath, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                c = p[3].strip() if len(p) > 3 else ""
                if c == country:
                    tid = p[0].strip()
                    country_targets[tid] = (normalize_text(p[1]), normalize_text(p[2]), extract_digits(p[2]))

    test_blocker = FastBlockingEngine(max_cands_per_s1=25)
    test_blocker.fit(country_targets)

    for sid, (sn, sa, s_dig) in s1_dict.items():
        cands = test_blocker.query(sn, sa, s_dig)
        all_candidates[sid] = cands
        if not cands: continue

        batch_feats = []
        valid_cids = []
        for cid in cands:
            if cid in country_targets:
                tn, ta, t_dig = country_targets[cid]
                batch_feats.append(extract_features(sn, sa, s_dig, tn, ta, t_dig))
                valid_cids.append(cid)

        if batch_feats:
            probs = model.predict(np.array(batch_feats))
            for cid, pr in zip(valid_cids, probs):
                all_scores[sid].append((cid, float(pr)))

    del country_targets
    gc.collect()

target_proposals = defaultdict(list)
for s1, cands in all_scores.items():
    for tid, score in cands:
        target_proposals[tid].append((s1, score))

target_owner = {}
for tid, props in target_proposals.items():
    best_s1, _ = max(props, key=lambda x: x[1])
    target_owner[tid] = best_s1

final_matches = {s1: [] for s1 in s1_order}
for s1 in s1_order:
    proposals = all_scores.get(s1, [])
    owned = [(tid, sc) for tid, sc in proposals if target_owner.get(tid) == s1]
    owned.sort(key=lambda x: x[1], reverse=True)

    selected = []
    for idx, (tid, score) in enumerate(owned):
        thresh = 0.65 if idx == 0 else 0.78
        if score >= thresh:
            selected.append(tid)
    final_matches[s1] = sorted(selected)

out_match = os.path.join(OUTPUT_DIR, "matching_results.tsv")
out_cand = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

with open(out_match, "w", encoding="utf-8") as fm, open(out_cand, "w", encoding="utf-8") as fc:
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in s1_order:
        fc.write(f"{sid}\t{','.join(all_candidates.get(sid, []))}\n")
        fm.write(f"{sid}\t{','.join(final_matches.get(sid, []))}\n")

print("\n" + "=" * 80)
print(f"SUCCESS! Output files updated in: {OUTPUT_DIR}")
print("=" * 80)
