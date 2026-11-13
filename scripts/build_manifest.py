"""Build a verified download manifest for offline installation and model caching."""

from __future__ import annotations

import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCKFILE = ROOT / "requirements.lock.txt"
WHEELS_DIR = ROOT / "wheels"
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = WHEELS_DIR / "manifest.json"

PYTORCH_CU130_INDEX = "https://download.pytorch.org/whl/cu130"
HF_MODEL_REPO = "facebook/dinov2-with-registers-small"
HF_MODEL_FILES = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
]


def _http_get_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-brain-setup/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-brain-setup/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_head_size(url: str, timeout: int = 20) -> int | None:
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "ai-brain-setup/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl is not None else None


def _pick_wheel_rank(filename: str) -> int | None:
    """Return preference rank (lower is better) for Python 3.12 win_amd64."""
    if not filename.endswith(".whl"):
        return None
    if "cp312-cp312-win_amd64" in filename:
        return 0
    if re.search(r"cp3\d+-abi3-win_amd64", filename):
        return 1
    if "py3-none-win_amd64" in filename:
        return 2
    if "py3-none-any" in filename:
        return 3
    if "py2.py3-none-any" in filename:
        return 4
    # Fallback for any cp3xx-none-any etc.
    if filename.endswith("-none-any.whl"):
        return 5
    return None


def _resolve_pytorch_wheel(pkg: str, version: str) -> dict:
    index_url = f"{PYTORCH_CU130_INDEX}/{pkg}/"
    page = _http_get_text(index_url)
    hrefs = re.findall(r'href="([^"]+)"', page)
    encoded_ver = urllib.parse.quote(version, safe="")
    candidates = []
    for raw_href in hrefs:
        href = html.unescape(raw_href)
        url_part, _, frag = href.partition("#")
        fname = urllib.parse.unquote(Path(urllib.parse.urlparse(url_part).path).name)
        if not (fname.startswith(f"{pkg}-{version}-") or f"{pkg}-{encoded_ver}-" in url_part):
            continue
        rank = _pick_wheel_rank(fname)
        if rank is None:
            continue
        sha256 = None
        if frag.startswith("sha256="):
            sha256 = frag.split("=", 1)[1].strip()
        full_url = urllib.parse.urljoin(index_url, url_part)
        candidates.append((rank, fname, full_url, sha256))
    if not candidates:
        raise RuntimeError(f"No cp312 win_amd64 wheel found for {pkg}=={version} at {index_url}")
    candidates.sort(key=lambda item: item[0])
    _, fname, full_url, sha256 = candidates[0]
    size = _http_head_size(full_url)
    return {
        "name": pkg,
        "version": version,
        "filename": fname,
        "url": full_url,
        "size": size,
        "sha256": sha256,
        "dest": str(WHEELS_DIR / fname),
    }


def _resolve_pypi_wheel(pkg: str, version: str) -> dict:
    url = f"https://pypi.org/pypi/{pkg}/{version}/json"
    data = _http_get_json(url)
    candidates = []
    for file_info in data.get("urls", []):
        if file_info.get("packagetype") != "bdist_wheel":
            continue
        fname = file_info["filename"]
        rank = _pick_wheel_rank(fname)
        if rank is None:
            continue
        candidates.append((rank, file_info))
    if not candidates:
        raise RuntimeError(f"No compatible wheel found on PyPI for {pkg}=={version}")
    candidates.sort(key=lambda item: item[0])
    chosen = candidates[0][1]
    return {
        "name": pkg,
        "version": version,
        "filename": chosen["filename"],
        "url": chosen["url"],
        "size": chosen.get("size"),
        "sha256": chosen.get("digests", {}).get("sha256"),
        "dest": str(WHEELS_DIR / chosen["filename"]),
    }


def _resolve_hf_model_files() -> list[dict]:
    items = []
    model_dir = MODELS_DIR / "dinov2-with-registers-small"
    for rel in HF_MODEL_FILES:
        file_url = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/{rel}"
        size = _http_head_size(file_url)
        items.append(
            {
                "name": f"{HF_MODEL_REPO}/{rel}",
                "filename": rel,
                "url": file_url,
                "size": size,
                "sha256": None,
                "dest": str(model_dir / rel),
            }
        )
    return items


def parse_lockfile(path: Path) -> list[tuple[str, str]]:
    pkgs = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        name, ver = line.split("==", 1)
        pkgs.append((name.strip(), ver.strip()))
    return pkgs


def main() -> int:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pkgs = parse_lockfile(LOCKFILE)
    wheels = []
    for pkg, ver in pkgs:
        if "+cu" in ver or pkg in {"torch", "torchvision", "torchaudio"} and "+" in ver:
            entry = _resolve_pytorch_wheel(pkg, ver)
        else:
            entry = _resolve_pypi_wheel(pkg, ver)
        wheels.append(entry)
        mb = (entry["size"] or 0) / (1024 * 1024)
        print(f"Resolved {entry['filename']} ({mb:.2f} MB)")

    # Download smallest wheels first so 57/58 packages complete rapidly before torch+cu130
    wheels.sort(key=lambda w: w.get("size") or 0)
    models = _resolve_hf_model_files()
    for m in models:
        mb = (m["size"] or 0) / (1024 * 1024)
        print(f"Resolved model {m['name']} ({mb:.2f} MB)")

    manifest = {
        "python": "3.12",
        "platform": "win_amd64",
        "wheels": wheels,
        "models": models,
        "total_wheels_bytes": sum(w.get("size") or 0 for w in wheels),
        "total_models_bytes": sum(m.get("size") or 0 for m in models),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total_mb = (manifest["total_wheels_bytes"] + manifest["total_models_bytes"]) / (1024 * 1024)
    print(f"\nManifest saved to {MANIFEST_PATH} ({len(wheels)} wheels + {len(models)} model files, {total_mb:.2f} MB total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
