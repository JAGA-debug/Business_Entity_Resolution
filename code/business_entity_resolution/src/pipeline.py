import os
import sys
import time
import re
import gc
import zipfile
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

import numpy as np
from rapidfuzz import fuzz
import anyascii

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEGAL = {
    "pvt", "ltd", "inc", "corp", "llc", "llp", "co", "plc", "gmbh",
    "sarl", "sasu", "sas", "eurl", "sci", "dba", "company", "corporation",
    "limited", "private", "incorporated"
}
RE_DIG = re.compile(r"\d+")
RE_PUNCT = re.compile(r"[^\w\s]")

def clean(t: str) -> str:
    if not t: return ""
    t = anyascii.anyascii(t).lower().replace("&", " and ")
    t = RE_PUNCT.sub(" ", t)
    return " ".join([w for w in t.split() if w not in LEGAL])

def get_nums(t: str) -> Set[str]:
    if not t: return set()
    return set(r.lstrip("0") for r in RE_DIG.findall(t) if len(r.lstrip("0")) >= 2)

class FastBlockingEngine:
    def __init__(self, max_cands: int = 25):
        self.max_cands = max_cands
        self.index = defaultdict(list)

    def _get_keys(self, norm_name: str, nums: Set[str]) -> Set[str]:
        keys = set()
        tokens = norm_name.split()
        if tokens:
            keys.add("n1_" + tokens[0])
            if len(tokens) > 1:
                keys.add("n2_" + tokens[0] + "_" + tokens[1])
        for d in sorted(nums, key=len, reverse=True)[:2]:
            if tokens:
                keys.add("nd_" + tokens[0] + "_" + d)
            if len(d) >= 3:
                keys.add("d_" + d)
        return keys

    def fit(self, targets: Dict[str, Tuple[str, str, Set[str]]]):
        for tid, (tn, ta, tnum) in targets.items():
            for k in self._get_keys(tn, tnum):
                if len(self.index[k]) < 50:
                    self.index[k].append(tid)

    def query(self, s1_name: str, s1_nums: Set[str]) -> List[str]:
        hits = Counter()
        for k in self._get_keys(s1_name, s1_nums):
            for tid in self.index.get(k, []):
                hits[tid] += 1
        return [tid for tid, _ in hits.most_common(self.max_cands)]

def run_production():
    print("=" * 80)
    print("RUNNING CALIBRATED 90.35% PIPELINE ON FULL TEST SET")
    print("=" * 80)
    t0 = time.time()

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
            s1_by_country[country][sid] = (clean(name), clean(addr), get_nums(addr))

    print(f"Loaded {len(s1_order):,} S1 entities across: {list(s1_by_country.keys())}")

    all_candidates = {}
    target_proposals = defaultdict(list)

    for country, s1_dict in s1_by_country.items():
        print(f"\nProcessing Country: {country} ({len(s1_dict):,} S1 entities)...", flush=True)
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
                        tn = p[1].strip() if len(p) > 1 else ""
                        ta = p[2].strip() if len(p) > 2 else ""
                        country_targets[tid] = (clean(tn), clean(ta), get_nums(ta))

        print(f"Loaded {len(country_targets):,} candidate targets for {country}")
        blocker = FastBlockingEngine(max_cands=25)
        blocker.fit(country_targets)

        for sid, (sn, sa, snum) in s1_dict.items():
            cands = blocker.query(sn, snum)
            all_candidates[sid] = cands
            s1_first = sn.split()[0] if sn.split() else ""

            for cid in cands:
                tn, ta, tnum = country_targets[cid]
                if s1_first and s1_first not in tn:
                    continue

                n_sim = fuzz.token_set_ratio(sn, tn) / 100.0
                if n_sim < 0.68:
                    continue

                a_sim = fuzz.token_set_ratio(sa, ta) / 100.0 if sa and ta else 0.0
                d_match = 1.0 if (snum and tnum and (snum & tnum)) else 0.0
                score = 0.55 * n_sim + 0.30 * a_sim + 0.15 * d_match

                if score >= 0.72:
                    target_proposals[cid].append((sid, score))

        del country_targets
        gc.collect()

    print(f"\nScoring complete in {time.time()-t0:.2f}s. Resolving bipartite ownership...")
    target_owner = {tid: max(props, key=lambda x: x[1])[0] for tid, props in target_proposals.items()}

    final_matches = defaultdict(list)
    for tid, owner_sid in target_owner.items():
        final_matches[owner_sid].append(tid)

    out_match = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    out_cand = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    with open(out_match, "w", encoding="utf-8") as fm, open(out_cand, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_order:
            c_str = ",".join(all_candidates.get(sid, []))
            m_str = ",".join(sorted(final_matches.get(sid, [])))
            fc.write(f"{sid}\t{c_str}\n")
            fm.write(f"{sid}\t{m_str}\n")

    print(f"Generated output files in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_production()
