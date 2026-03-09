from pathlib import Path
from datetime import datetime
import json
import hashlib
import yaml

def run_id_from_cfg(cfg: dict) -> str:
    """
    Build a deterministic short run id from key config fields.
    Why: same (seed, policy, data path) -> same id, so outputs are organized.
    """
    key = f"{cfg.get('seed')}|{cfg.get('policy')}|{cfg.get('data',{}).get('prices_csv')}"
    return hashlib.sha1(key.encode()).hexdigest()[:8]

def prepare_outdir(base: str, run_id: str) -> Path:
    """
    Create outputs/runs/<run_id>/ and a figures/ subfolder.
    Why: every run writes in its own folder, easy to share and compare.
    """
    out = Path(base).parent / "runs" / run_id
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out

def snapshot(out_dir: Path, cfg_path="configs/run.yaml", costs_path="configs/costs.yaml"):
    """
    Write a run.json with exact configs and timestamp.
    Why: future you (and teammates) can reproduce this run byte-for-byte.
    """
    info = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "run_yaml": yaml.safe_load(Path(cfg_path).read_text()),
        "costs_yaml": yaml.safe_load(Path(costs_path).read_text()),
    }
    (out_dir / "run.json").write_text(json.dumps(info, indent=2))
