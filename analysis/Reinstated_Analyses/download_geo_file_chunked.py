from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


requests.packages.urllib3.disable_warnings()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("target")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-kb", type=int, default=256)
    args = parser.parse_args()
    target = Path(args.target)
    marker_dir = target.parent / f"{target.name}.chunks"
    marker_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunk_kb * 1024
    head = requests.head(args.url, allow_redirects=True, timeout=60, verify=False)
    head.raise_for_status()
    total = int(head.headers["Content-Length"])
    if target.exists() and target.stat().st_size == total and not marker_dir.exists():
        print("already complete")
        return
    with target.open("wb") as handle:
        handle.truncate(total)
    n_chunks = (total + chunk_size - 1) // chunk_size

    def bounds(index: int) -> tuple[int, int]:
        start = index * chunk_size
        return start, min(start + chunk_size - 1, total - 1)

    def marker(index: int) -> Path:
        return marker_dir / f"{index:05d}.json"

    def complete(index: int) -> bool:
        path = marker(index)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            start, end = bounds(index)
            return payload.get("size") == end - start + 1
        except Exception:
            return False

    def fetch(index: int) -> int:
        if complete(index):
            return index
        start, end = bounds(index)
        expected = end - start + 1
        last_error = ""
        for attempt in range(1, 101):
            try:
                response = requests.get(
                    args.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(30, 240),
                    stream=True,
                    verify=False,
                )
                response.raise_for_status()
                data = bytearray()
                for piece in response.iter_content(64 * 1024):
                    if piece:
                        data.extend(piece)
                if len(data) != expected:
                    raise OSError(f"expected {expected}, got {len(data)}")
                with target.open("r+b") as handle:
                    handle.seek(start)
                    handle.write(data)
                marker(index).write_text(
                    json.dumps(
                        {
                            "index": index,
                            "size": expected,
                            "sha256": hashlib.sha256(data).hexdigest(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return index
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(3 * attempt, 60))
        raise RuntimeError(f"chunk {index} failed: {last_error}")

    completed = sum(complete(i) for i in range(n_chunks))
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(fetch, i) for i in range(n_chunks) if not complete(i)
        ]
        for future in as_completed(futures):
            future.result()
            completed = sum(complete(i) for i in range(n_chunks))
            if completed % max(1, n_chunks // 20) == 0 or completed == n_chunks:
                elapsed = max(time.time() - started, 1)
                print(
                    f"{completed}/{n_chunks} chunks; "
                    f"{completed * chunk_size / elapsed:.0f} bytes/s",
                    flush=True,
                )
    if target.stat().st_size != total:
        raise SystemExit("size validation failed")
    print("complete", target)


if __name__ == "__main__":
    main()
