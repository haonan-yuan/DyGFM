#!/usr/bin/env python3
"""
Verify DyGFM initial data and build static/data/*.pt when possible.

Required inputs (copy from release bundle or node-time):
  processed/ml_{ds}.npy, processed/ml_{ds}_node.npy
  graph_structure/{ds}_edge_index.npy
  downstream_data/{ds}/ds_{ds}.csv  (+ genre: ds_genre_{1..5}.csv)
  dynamic/data/features/{ds}.pt
  dynamic/data/normal_time_pt/{ds}.pt
  dynamic/data/edge_feature/{ds}.pt

Generated:
  static/data/{ds}.pt  (from rebuild_static_pt)
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import (
    DATASETS,
    DOWNSTREAM_DATA_DIR,
    DYNAMIC_EDGE_FEATURE_DIR,
    DYNAMIC_FEATURES_DIR,
    DYNAMIC_NORMAL_TIME_DIR,
    GRAPH_STRUCTURE_DIR,
    PROCESSED_DIR,
    STATIC_DATA_DIR,
    rel,
)


def _check(path: str) -> bool:
    ok = os.path.isfile(path)
    mark = "OK" if ok else "MISSING"
    print(f"  [{mark}] {rel(path)}")
    return ok


def required_files(dataset: str) -> list:
    files = [
        os.path.join(PROCESSED_DIR, f"ml_{dataset}_node.npy"),
        os.path.join(PROCESSED_DIR, f"ml_{dataset}.npy"),
        os.path.join(GRAPH_STRUCTURE_DIR, f"{dataset}_edge_index.npy"),
        os.path.join(DOWNSTREAM_DATA_DIR, dataset, f"ds_{dataset}.csv"),
        os.path.join(DYNAMIC_FEATURES_DIR, f"{dataset}.pt"),
        os.path.join(DYNAMIC_NORMAL_TIME_DIR, f"{dataset}.pt"),
        os.path.join(DYNAMIC_EDGE_FEATURE_DIR, f"{dataset}.pt"),
    ]
    if dataset == "genre":
        for k in range(1, 6):
            files.append(
                os.path.join(
                    DOWNSTREAM_DATA_DIR, "genre", f"ds_genre_{k}.csv"
                )
            )
    return files


def main():
    print(f"DyGFM data check (root={PROJECT_ROOT})\n")
    all_ok = True
    for ds in DATASETS:
        print(f"== {ds} ==")
        for path in required_files(ds):
            all_ok &= _check(path)

    print("\n== static/data/*.pt (build if missing) ==")
    import importlib.util

    rebuild_path = os.path.join(PROJECT_ROOT, "static", "rebuild_static_pt.py")
    spec = importlib.util.spec_from_file_location("rebuild_static_pt", rebuild_path)
    rebuild_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rebuild_mod)
    rebuild = rebuild_mod.rebuild

    for ds in DATASETS:
        out = os.path.join(STATIC_DATA_DIR, f"{ds}.pt")
        node_npy = os.path.join(PROCESSED_DIR, f"ml_{ds}_node.npy")
        edge_npy = os.path.join(GRAPH_STRUCTURE_DIR, f"{ds}_edge_index.npy")
        if os.path.isfile(out):
            print(f"  [OK] {rel(out)}")
            continue
        if os.path.isfile(node_npy) and os.path.isfile(edge_npy):
            try:
                rebuild(ds)
            except Exception as e:
                print(f"  [FAIL] rebuild {ds}: {e}")
                all_ok = False
        else:
            print(f"  [MISSING] {rel(out)} (need processed + graph_structure)")
            all_ok = False

    if all_ok:
        print("\nAll required data present.")
        return 0
    print("\nSome files are missing. Copy initial data into DyGFM/ then re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
