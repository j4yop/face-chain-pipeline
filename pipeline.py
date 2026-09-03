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
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/"
    "Cristiano_Ronaldo%2C_2010.jpg/500px-Cristiano_Ronaldo%2C_2010.jpg"
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
    """Reverse-image search.

    Provider order (controlled by .env):
      1. SERPAPI_KEY  -> Google Lens via SerpAPI (no card, 250/mo free)
      2. GOOGLE_VISION_API_KEY  -> Google Cloud Vision WEB_DETECTION
      3. GOOGLE_APPLICATION_CREDENTIALS  -> Vision via service account

    Both providers are normalized into a single shape with four buckets:
        pages_with_matching_images, full_matching_images,
        partial_matching_images, visually_similar_images
    plus best_guess_labels, so the picker logic is identical.
    """
    if os.environ.get("SERPAPI_KEY"):
        return _serpapi_lens(img_path)
    return _google_vision(img_path)


def _google_vision(img_path: Path) -> dict[str, Any]:
    api_key = os.environ.get("GOOGLE_VISION_API_KEY")
    if api_key:
        client = vision.ImageAnnotatorClient(client_options={"api_key": api_key})
    else:
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


def _serpapi_lens(img_path: Path) -> dict[str, Any]:
    """Reverse-image via SerpAPI Google Lens. Free tier: 250 searches/month, no card."""
    api_key = os.environ["SERPAPI_KEY"]
    # 1) Upload the image (valid 10 min).
    with open(img_path, "rb") as f:
        up = requests.post(
            "https://serpapi.com/image",
            files={"image": (img_path.name, f, "image/jpeg")},
            data={"api_key": api_key},
            timeout=30,
        )
    up.raise_for_status()
    up_data = up.json()
    if "image_id" not in up_data:
        raise RuntimeError(f"SerpAPI upload failed: {up_data}")
    image_id = up_data["image_id"]

    # 2) Query Google Lens across multiple result buckets. SerpAPI's
    #    `exact_matches` often wraps every result in opaque Google goto
    #    redirects; `visual_matches` and `organic_results` frequently
    #    return direct social-media URLs. We merge from all three and
    #    de-duplicate, preferring direct (non-google.com) URLs.
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for type_ in ("exact_matches", "visual_matches", "organic_results"):
        params = {
            "engine": "google_lens",
            "image_id": image_id,
            "type": type_,
            "api_key": api_key,
        }
        try:
            res = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
            res.raise_for_status()
            data = res.json()
        except Exception:
            continue
        if "error" in data:
            continue
        bucket = data.get(type_) or []
        for m in bucket:
            url = m.get("link") or m.get("redirect_link") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            candidates.append({
                "url": url,
                "type": type_,
                "title": m.get("title", ""),
                "source": m.get("source", "") or m.get("source_domain", ""),
            })

    # Normalize: all candidates go into all four buckets so the picker can
    # find them. The picker itself does the unwrap-redirect work.
    return {
        "pages_with_matching_images": [{"url": c["url"], "score": 1.0, "title": c["title"], "source": c["source"]} for c in candidates],
        "full_matching_images":       [{"url": c["url"], "score": 1.0, "title": c["title"], "source": c["source"]} for c in candidates],
        "partial_matching_images":    [{"url": c["url"], "score": 0.5, "title": c["title"], "source": c["source"]} for c in candidates],
        "visually_similar_images":    [{"url": c["url"], "title": c["title"], "source": c["source"]}             for c in candidates],
        "best_guess_labels":          [],
    }


def follow_redirects(url: str, timeout: int = 10) -> tuple[str, list[str]]:
    """Follow HTTP redirects (with browser-like headers) and return
    (final_url, redirect_chain). Returns (url, []) if no redirect or any
    non-fatal error. Used to unwrap Google 'goto' and 'url' links."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        chain = [resp.url for resp in r.history] + [r.url]
        r.close()
        return r.url, chain
    except Exception:
        return url, []


def unwrap_redirect(url: str, follow: bool = True) -> str:
    """Google wraps real URLs in https://www.google.com/url?... or
    https://www.google.com/goto?url=CAES... — try to extract the real one.

    The 'url' query value may itself be a base64-encoded Google internal
    payload, in which case we follow the HTTP redirect to get the real URL.
    """
    if "google.com/url?" in url or "google.com/goto?" in url or "google.com/search?" in url:
        from urllib.parse import urlparse, parse_qs
        try:
            q = parse_qs(urlparse(url).query)
            for key in ("url", "q", "imgurl"):
                if key in q and q[key]:
                    inner = q[key][0]
                    if follow and inner.startswith(("CAES", "CAUY", "http")):
                        final, _ = follow_redirects(inner)
                        if "google.com" not in final:
                            return final
                        return inner
                    return inner
        except Exception:
            pass
    return url


def is_social_text(s: str) -> bool:
    """Check if a URL or human-readable title contains a social-host hint."""
    s_low = s.lower()
    return any(h in s_low for h in SOCIAL_HOSTS) or any(
        kw in s_low for kw in ("facebook", "twitter", "reddit", "instagram", "tiktok", "youtube", "linkedin", "pinterest", "tumblr")
    )


def pick_best_social_match(web: dict[str, Any]) -> dict[str, Any] | None:
    """Walk pages -> full -> partial -> similar, return first social-host URL.
    Unwrap Google redirectors (and follow them when possible). Strongly
    prefer direct social URLs over indirect ones (Google goto wrappers
    whose destination is unreachable from Python)."""
    collected: list[dict[str, Any]] = []
    for bucket, key in [
        ("pages_with_matching_images", "url"),
        ("full_matching_images", "url"),
        ("partial_matching_images", "url"),
        ("visually_similar_images", "url"),
    ]:
        for entry in web.get(bucket, []):
            raw = entry.get(key, "")
            title = entry.get("title", "")
            source = entry.get("source", "")
            url = unwrap_redirect(raw) if raw else ""
            collected.append({
                "url": url or raw, "raw_url": raw, "title": title, "source": source, "bucket": bucket,
            })

    # Rank: direct social URL > direct non-social URL > Google-wrapper with social title/source > anything.
    def rank(c: dict[str, Any]) -> tuple[int, int]:
        u = c["url"] or ""
        is_direct_social = is_social_text(u) and "google.com" not in u
        title_or_source_social = (not is_direct_social) and is_social_text(f"{c['title']} {c['source']}")
        return (
            0 if is_direct_social else (1 if title_or_source_social else 2),
            0 if is_direct_social else 1,
        )

    social = [c for c in collected if is_social_text(c["url"]) or is_social_text(f"{c['title']} {c['source']}")]
    if not social:
        return None
    social.sort(key=rank)
    best = social[0]
    best["title_or_source_only"] = (not is_social_text(best["url"]) and
                                    is_social_text(f"{best['title']} {best['source']}"))
    return best


# ---------- Stage 3: blockchain anchor -------------------------------------

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _connect_web3() -> Web3:
    """Connect to Sepolia. Try user's RPC, then public fallbacks."""
    rpcs = [os.environ["SEPOLIA_RPC_URL"]] + [
        "https://ethereum-sepolia-rpc.publicnode.com",
        "https://rpc.sepolia.org",
        "https://1rpc.io/sepolia",
    ]
    last_err = None
    for rpc in rpcs:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 15}))
            if w3.is_connected():
                # Sanity: get a real block number.
                bn = w3.eth.block_number
                if bn > 0:
                    return w3
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Sepolia RPC unreachable. Tried: {rpcs}. Last error: {last_err}. "
        f"Check SEPOLIA_RPC_URL in .env — your Alchemy key may be invalid."
    )


def _gas_price_oracle(w3: Web3) -> int:
    """Sepolia miners reject sub-5 gwei in 2026. Use max of network's
    gas price and a 5 gwei floor."""
    floor = 5_000_000_000
    try:
        suggested = w3.eth.gas_price
    except Exception:
        return floor
    return max(suggested, floor)


def anchor_on_sepolia(payload: dict[str, Any]) -> dict[str, Any]:
    w3 = _connect_web3()
    pk = os.environ["SEPOLIA_PRIVATE_KEY"]
    acct = Account.from_key(pk)
    payload_bytes = json.dumps(payload, sort_keys=True).encode()
    payload_hex = "0x" + payload_bytes.hex()

    gas_price = _gas_price_oracle(w3)
    tx = {
        "to": acct.address,
        "from": acct.address,
        "value": 0,
        "data": payload_hex,
        "gas": 200_000,
        "gasPrice": gas_price,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "chainId": SEPOLIA_CHAIN_ID,
    }
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"    Sent tx with gasPrice={gas_price / 1e9:.2f} gwei, nonce={tx['nonce']}")
    print(f"    Tx hash: {tx_hash.hex()}")

    try:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180, poll_latency=5)
    except Exception as e:
        # Tx may have mined but our RPC missed it. Try fetching the receipt by hash from a fresh RPC.
        print(f"    Primary RPC timed out waiting. Falling back to Etherscan-style lookup...")
        try:
            tx_block = w3.eth.get_transaction(tx_hash).blockNumber
            if tx_block:
                receipt = w3.eth.get_transaction_receipt(tx_hash)
            else:
                raise e
        except Exception:
            raise

    return {
        "tx_hash": tx_hash.hex(),
        "block_number": receipt.blockNumber,
        "block_hash": receipt.blockHash.hex(),
        "gas_used": receipt.gasUsed,
        "gas_price_gwei": gas_price / 1e9,
        "etherscan_url": ETHERSCAN_BASE + tx_hash.hex(),
    }


# ---------- Orchestration --------------------------------------------------

def run(image_arg: str | None) -> dict[str, Any]:
    load_dotenv()
    has_image_search = (
        os.environ.get("SERPAPI_KEY")
        or os.environ.get("GOOGLE_VISION_API_KEY")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if not has_image_search:
        sys.exit("Missing reverse-image search key. Set SERPAPI_KEY (recommended, no card) "
                 "or GOOGLE_VISION_API_KEY / GOOGLE_APPLICATION_CREDENTIALS in .env")
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
    provider = "SerpAPI (Google Lens)" if os.environ.get("SERPAPI_KEY") else "Google Cloud Vision WEB_DETECTION"
    print(f"\n[2/4] Running reverse-image search via {provider}...")
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
            "provider": "SerpAPI (Google Lens)" if os.environ.get("SERPAPI_KEY") else "Google Cloud Vision",
            "best_guess_labels": web["best_guess_labels"][:5],
            "n_pages_with_matching_images": len(web["pages_with_matching_images"]),
            "n_full_matching_images": len(web["full_matching_images"]),
            "n_partial_matching_images": len(web["partial_matching_images"]),
            "n_visually_similar_images": len(web["visually_similar_images"]),
        },
        "reverse_image_provider": "SerpAPI (Google Lens)" if os.environ.get("SERPAPI_KEY") else "Google Cloud Vision",
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
        f"REVERSE-IMAGE SEARCH ({'SerpAPI Google Lens' if os.environ.get('SERPAPI_KEY') else 'Google Cloud Vision WEB_DETECTION'})",
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
