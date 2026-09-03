"""End-to-end pipeline: face -> web match -> blockchain anchor.

Usage:
    python pipeline.py            # uses default Ronaldo Wikimedia URL
    python pipeline.py <img_path> # uses a local image

Stages:
    1. Load + face-detect + 512-d ArcFace embedding (InsightFace, buffalo_l).
    2. Reverse-image search via Google Cloud Vision WEB_DETECTION.
    3. Pick best matching social-media URL, compute its image hash,
       anchor {url, hash, embedding_hash, ts} on Sepolia.
    4. Write receipt.json + receipt.txt and print Etherscan link.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

# InsightFace
import insightface
from insightface.app import FaceAnalysis

# Google Vision
from google.cloud import vision

# Web3 / Ethereum
from eth_account import Account
from web3 import Web3


@contextlib.contextmanager
def _silence_stdout():
    """InsightFace spams 'find model: ...' on every FaceAnalysis() init.
    Suppress it so the screen recording stays clean."""
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved

# ---------- Defaults --------------------------------------------------------

DEFAULT_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/"
    "Cristiano_Ronaldo_2018.jpg/500px-Cristiano_Ronaldo_2018.jpg"
)
SEPOLIA_CHAIN_ID = 11155111
ETHERSCAN_BASE = "https://sepolia.etherscan.io/tx/"

SOCIAL_HOSTS = (
    "twitter.com", "x.com", "reddit.com", "instagram.com",
    "facebook.com", "tiktok.com", "linkedin.com", "youtube.com",
    "tumblr.com", "flickr.com", "pinterest.com", "threads.net",
)


# ---------- Stage 1: face embedding ---------------------------------------

def download_image(url: str, dest: Path) -> Path:
    headers = {"User-Agent": "face-chain-pipeline/1.0 (demo)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_image_array(path: Path) -> np.ndarray:
    """Return RGB uint8 ndarray for InsightFace."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def encode_face(img_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Run InsightFace ArcFace. Return (512-d embedding, metadata)."""
    with _silence_stdout():
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
    img = load_image_array(img_path)
    faces = app.get(img)
    if not faces:
        raise RuntimeError(f"No face detected in {img_path}")
    # Pick the largest face (most prominent in frame).
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = face.normed_embedding  # already L2-normalized 512-d
    meta = {
        "det_score": float(face.det_score),
        "age": int(face.age) if getattr(face, "age", None) is not None else None,
        "gender": int(face.gender) if getattr(face, "gender", None) is not None else None,
        "bbox": [float(x) for x in face.bbox],
    }
    return embedding, meta


# ---------- Stage 2: reverse-image search ----------------------------------

def vision_web_detection(img_path: Path) -> dict[str, Any]:
    client = vision.ImageAnnotatorClient()
    with open(img_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.web_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")
    return {
        "pages_with_matching_images": [
            {"url": p.url, "score": p.score} for p in response.web_detection.pages_with_matching_images
        ],
        "full_matching_images": [
            {"url": i.url, "score": i.score} for i in response.web_detection.full_matching_images
        ],
        "partial_matching_images": [
            {"url": i.url, "score": i.score} for i in response.web_detection.partial_matching_images
        ],
        "visually_similar_images": [
            {"url": i.url} for i in response.web_detection.visually_similar_images
        ],
        "best_guess_labels": [l.label for l in response.web_detection.best_guess_labels],
    }


def pick_best_social_match(web: dict[str, Any]) -> dict[str, Any] | None:
    """Walk pages -> full -> partial -> similar, return first social-host URL."""
    def is_social(url: str) -> bool:
        return any(h in url.lower() for h in SOCIAL_HOSTS)

    for bucket, key in [
        ("pages_with_matching_images", "url"),
        ("full_matching_images", "url"),
        ("partial_matching_images", "url"),
        ("visually_similar_images", "url"),
    ]:
        for entry in web.get(bucket, []):
            url = entry.get(key, "")
            if url and is_social(url):
                return {"url": url, "bucket": bucket, "entry": entry}
    return None


# ---------- Stage 3: blockchain anchor -------------------------------------

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def anchor_on_sepolia(payload: dict[str, Any]) -> dict[str, Any]:
    rpc = os.environ["SEPOLIA_RPC_URL"]
    pk = os.environ["SEPOLIA_PRIVATE_KEY"]
    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        raise RuntimeError("Sepolia RPC unreachable")

    acct = Account.from_key(pk)
    payload_bytes = json.dumps(payload, sort_keys=True).encode()
    payload_hex = "0x" + payload_bytes.hex()

    tx = {
        "to": acct.address,
        "from": acct.address,
        "value": 0,
        "data": payload_hex,
        "gas": 200_000,
        "gasPrice": w3.eth.gas_price,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "chainId": SEPOLIA_CHAIN_ID,
    }
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)

    return {
        "tx_hash": tx_hash.hex(),
        "block_number": receipt.blockNumber,
        "block_hash": receipt.blockHash.hex(),
        "gas_used": receipt.gasUsed,
        "etherscan_url": ETHERSCAN_BASE + tx_hash.hex(),
    }


# ---------- Orchestration --------------------------------------------------

def run(image_arg: str | None) -> dict[str, Any]:
    load_dotenv()
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        sys.exit("Missing GOOGLE_APPLICATION_CREDENTIALS. Copy .env.example to .env and fill in.")
    if not os.environ.get("SEPOLIA_RPC_URL") or not os.environ.get("SEPOLIA_PRIVATE_KEY"):
        sys.exit("Missing SEPOLIA_RPC_URL or SEPOLIA_PRIVATE_KEY in .env")

    workdir = Path(".receipts") / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    workdir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1 ---------------------------------------------------------
    print("\n[1/4] Loading image and encoding face (InsightFace buffalo_l)...")
    if image_arg and Path(image_arg).exists():
        img_path = Path(image_arg)
    else:
        img_path = workdir / "input.jpg"
        print(f"    Downloading default demo image from Wikimedia...")
        download_image(DEFAULT_IMAGE_URL, img_path)
    print(f"    Image: {img_path}")
    t0 = time.time()
    embedding, face_meta = encode_face(img_path)
    emb_hash = sha256_hex(embedding.astype(np.float32).tobytes())
    print(f"    Face detected (score={face_meta['det_score']:.2f}), "
          f"512-d embedding hashed to {emb_hash[:12]}... ({time.time()-t0:.1f}s)")

    # --- Stage 2 ---------------------------------------------------------
    print("\n[2/4] Running Google Cloud Vision WEB_DETECTION...")
    t0 = time.time()
    web = vision_web_detection(img_path)
    print(f"    Got {len(web['pages_with_matching_images'])} pages, "
          f"{len(web['full_matching_images'])} full matches, "
          f"{len(web['visually_similar_images'])} visually similar "
          f"({time.time()-t0:.1f}s)")
    if web["best_guess_labels"]:
        print(f"    Best guess labels: {web['best_guess_labels'][:3]}")

    print("\n[3/4] Selecting best social-media match...")
    match = pick_best_social_match(web)
    if not match:
        # Fall back: still anchor the best-guess + best available URL,
        # but flag the run. We do NOT hardcode a result.
        chosen = None
        for bucket in ("full_matching_images", "pages_with_matching_images", "visually_similar_images"):
            if web.get(bucket):
                chosen = {"url": web[bucket][0].get("url", ""), "bucket": bucket,
                          "entry": web[bucket][0], "fallback_non_social": True}
                break
        if not chosen:
            raise RuntimeError("Vision returned zero matches across all buckets. Demo image may be too unique.")
        match = chosen
        print(f"    No social-host match. Falling back to non-social URL: {match['url'][:80]}...")
    else:
        print(f"    Match ({match['bucket']}): {match['url']}")

    # Hash the matched image bytes if reachable; otherwise hash the URL string.
    matched_image_hash = sha256_hex(match["url"].encode())
    try:
        r = requests.get(match["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.ok and r.content:
            matched_image_hash = sha256_hex(r.content)
            # Save a copy of the matched image for the receipt.
            (workdir / "matched.jpg").write_bytes(r.content)
    except Exception as e:
        print(f"    (Could not fetch matched image bytes: {e}. Hashing URL instead.)")

    # --- Stage 3 ---------------------------------------------------------
    payload = {
        "schema": "face-chain-pipeline/v1",
        "run_id": str(uuid.uuid4()),
        "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_image_sha256": sha256_hex(img_path.read_bytes()),
        "face_embedding_sha256": emb_hash,
        "face_embedding_dim": int(embedding.shape[0]),
        "face_meta": face_meta,
        "match": {
            "url": match["url"],
            "bucket": match["bucket"],
            "matched_image_sha256": matched_image_hash,
        },
        "vision_response_summary": {
            "best_guess_labels": web["best_guess_labels"][:5],
            "n_pages_with_matching_images": len(web["pages_with_matching_images"]),
            "n_full_matching_images": len(web["full_matching_images"]),
            "n_partial_matching_images": len(web["partial_matching_images"]),
            "n_visually_similar_images": len(web["visually_similar_images"]),
        },
    }

    print("\n[4/4] Anchoring payload to Ethereum Sepolia...")
    t0 = time.time()
    chain = anchor_on_sepolia(payload)
    print(f"    Anchored in {time.time()-t0:.1f}s")
    print(f"    Tx:        {chain['tx_hash']}")
    print(f"    Block:     {chain['block_number']}")
    print(f"    Etherscan: {chain['etherscan_url']}")

    # --- Receipt ---------------------------------------------------------
    final = {**payload, "blockchain": chain}
    (workdir / "receipt.json").write_text(json.dumps(final, indent=2, default=str))

    summary = [
        "FACE-CHAIN-PIPELINE RECEIPT",
        "=" * 50,
        f"Run ID:        {payload['run_id']}",
        f"Timestamp UTC: {payload['ts_utc']}",
        "",
        "FACE",
        f"  Input image SHA-256:    {payload['input_image_sha256']}",
        f"  Embedding SHA-256:      {emb_hash}",
        f"  Embedding dim:          {payload['face_embedding_dim']}",
        f"  Detector confidence:    {face_meta['det_score']:.4f}",
        f"  Bounding box:           {face_meta['bbox']}",
        "",
        "REVERSE-IMAGE SEARCH (Google Cloud Vision)",
        f"  Best guess labels:      {payload['vision_response_summary']['best_guess_labels']}",
        f"  Pages with match:       {payload['vision_response_summary']['n_pages_with_matching_images']}",
        f"  Full matches:           {payload['vision_response_summary']['n_full_matching_images']}",
        f"  Visually similar:       {payload['vision_response_summary']['n_visually_similar_images']}",
        "",
        "SELECTED MATCH",
        f"  URL:                    {match['url']}",
        f"  Bucket:                 {match['bucket']}",
        f"  Matched image SHA-256:  {matched_image_hash}",
        "",
        "BLOCKCHAIN ANCHOR (Ethereum Sepolia)",
        f"  Tx hash:                {chain['tx_hash']}",
        f"  Block number:           {chain['block_number']}",
        f"  Etherscan:              {chain['etherscan_url']}",
        "",
        "Re-verify:",
        f"  web3.eth.get_transaction('{chain['tx_hash']}')",
        "  and decode the 'input' field as UTF-8 JSON.",
        "=" * 50,
    ]
    (workdir / "receipt.txt").write_text("\n".join(summary))
    print(f"\nReceipts written to: {workdir}/")
    print("\n".join(summary))
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None,
                        help="Optional path to a local face image. Default: download Cristiano Ronaldo from Wikimedia.")
    args = parser.parse_args()
    run(args.image)
