"""Drop-proof, IP-change-proof downloader for wheels and models using curl -C -."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "wheels" / "manifest.json"
PROGRESS_PATH = ROOT / "wheels" / "progress.json"
CURL_EXE = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "curl.exe"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact(path: Path, expected_size: int | None, expected_sha256: str | None) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        return False, f"size mismatch ({actual_size} != {expected_size})"
    if expected_sha256:
        actual_sha = sha256_file(path)
        if actual_sha.lower() != expected_sha256.lower():
            return False, f"sha256 mismatch ({actual_sha[:12]} != {expected_sha256[:12]})"
    if path.name.endswith(".whl") or path.name.endswith(".whl.part"):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    return False, f"corrupt zip member: {bad}"
        except zipfile.BadZipFile as exc:
            return False, f"bad zip: {exc}"
    if path.name.endswith(".safetensors") or path.name.endswith(".safetensors.part"):
        try:
            with path.open("rb") as f:
                hdr_len = struct.unpack("<Q", f.read(8))[0]
                if hdr_len <= 0 or hdr_len > 32 * 1024 * 1024:
                    return False, f"invalid safetensors header length: {hdr_len}"
                json.loads(f.read(hdr_len).decode("utf-8"))
        except Exception as exc:
            return False, f"invalid safetensors header: {exc}"
    return True, "ok"


def update_progress(completed: list[str], current: str | None, total_items: int, bytes_done: int, bytes_total: int) -> None:
    payload = {
        "completed_count": len(completed),
        "total_items": total_items,
        "current_item": current,
        "bytes_done": bytes_done,
        "bytes_total": bytes_total,
        "percent": round((bytes_done / bytes_total * 100.0) if bytes_total > 0 else 0.0, 2),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_one(item: dict, max_retries: int = 200, retry_wait: int = 5) -> None:
    dest = Path(item["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    expected_size = item.get("size")
    expected_sha256 = item.get("sha256")
    url = item["url"]

    ok, _ = verify_artifact(dest, expected_size, expected_sha256)
    if ok:
        print(f"[SKIP] {dest.name} (already verified)", flush=True)
        return

    if dest.exists():
        print(f"[WARN] Existing {dest.name} failed verification; re-downloading", flush=True)
        dest.unlink()

    curl_cmd_base = str(CURL_EXE if CURL_EXE.exists() else "curl.exe")
    for attempt in range(1, max_retries + 1):
        if part.exists() and expected_size is not None:
            psize = part.stat().st_size
            if psize == expected_size:
                ok_part, reason = verify_artifact(part, expected_size, expected_sha256)
                if ok_part:
                    part.replace(dest)
                    print(f"[OK] {dest.name} verified from existing .part", flush=True)
                    return
                print(f"[WARN] .part size matched but failed check ({reason}); resetting", flush=True)
                part.unlink()
            elif psize > expected_size:
                print(f"[WARN] .part larger than expected ({psize} > {expected_size}); resetting", flush=True)
                part.unlink()

        offset_mb = (part.stat().st_size / (1024 * 1024)) if part.exists() else 0.0
        total_mb = (expected_size / (1024 * 1024)) if expected_size else 0.0
        print(
            f"[GET attempt {attempt}/{max_retries}] {dest.name} (resuming at {offset_mb:.2f}/{total_mb:.2f} MB)",
            flush=True,
        )

        cmd = [
            curl_cmd_base,
            "-C", "-",
            "-L",
            "--fail",
            "--connect-timeout", "15",
            "--speed-limit", "2048",
            "--speed-time", "15",
            "--silent",
            "--show-error",
            "-o", str(part),
            url,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            ok_part, reason = verify_artifact(part, expected_size, expected_sha256)
            if ok_part:
                part.replace(dest)
                print(f"[OK] {dest.name} ({total_mb:.2f} MB) verified", flush=True)
                return
            print(f"[WARN] Verification failed for {dest.name}: {reason}; resetting .part", flush=True)
            if part.exists():
                part.unlink()
        else:
            err_msg = (res.stderr or "").strip().splitlines()[-1:] or ["network error"]
            print(
                f"[WAIT] Interrupted ({res.returncode}: {err_msg[0]}). Waiting {retry_wait}s for reconnect/IP rotation...",
                flush=True,
            )
            # If server rejected range (416 or curl exit 33), validate or reset .part
            if res.returncode == 33 or "416" in (res.stderr or ""):
                ok_part, _ = verify_artifact(part, expected_size, expected_sha256)
                if ok_part:
                    part.replace(dest)
                    return
                if part.exists():
                    part.unlink()

        time.sleep(retry_wait)

    raise RuntimeError(f"Exceeded {max_retries} retries downloading {dest.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resilient drop-proof downloader")
    parser.add_argument("--skip-models", action="store_true", help="Download Python wheels only")
    parser.add_argument("--skip-torch", action="store_true", help="Skip large torch wheel")
    parser.add_argument("--max-retries", type=int, default=200)
    parser.add_argument("--retry-wait", type=int, default=5)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    items = list(manifest["wheels"])
    if not args.skip_models:
        items.extend(manifest["models"])
    if args.skip_torch:
        items = [x for x in items if x["name"] != "torch"]
    # Download smaller artifacts first so everything except torch+cu130 completes immediately
    items.sort(key=lambda x: x.get("size") or 0)

    bytes_total = sum(x.get("size") or 0 for x in items)
    completed: list[str] = []
    bytes_done = 0

    for item in items:
        update_progress(completed, item["filename"], len(items), bytes_done, bytes_total)
        download_one(item, max_retries=args.max_retries, retry_wait=args.retry_wait)
        completed.append(item["filename"])
        bytes_done += item.get("size") or 0
        update_progress(completed, None, len(items), bytes_done, bytes_total)

    print(f"\nALL {len(items)} ARTIFACTS DOWNLOADED & VERIFIED!", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
