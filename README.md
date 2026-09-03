# face-chain-pipeline

End-to-end pipeline: **face scan → reverse-image search → blockchain anchor**.

A single `python pipeline.py` run does all four stages and writes a verifiable
on-chain receipt on Ethereum Sepolia.

**Status:** ✅ Verified end-to-end on Sepolia. See [Verified run](#verified-run) below for the on-chain tx and screen recording.

---

## Verified run

A complete end-to-end run was performed on **2026-09-03** against the
public Ethereum Sepolia testnet.

| Field | Value |
|---|---|
| Demo subject | Public Wikipedia photo of Cristiano Ronaldo (CC-licensed) |
| Reverse-image provider | SerpAPI Google Lens (free tier, 250/mo) |
| Selected social-media match | [`https://www.instagram.com/p/DZtLoP-CFLR/`](https://www.instagram.com/p/DZtLoP-CFLR/) (real Instagram post) |
| On-chain transaction | [`0xdae079a311874e01046e0b2f9c324bf2bba269b53b70dc797ad9feb357bdbfeb`](https://sepolia.etherscan.io/tx/0xdae079a311874e01046e0b2f9c324bf2bba269b53b70dc797ad9feb357bdbfeb) |
| Block | `11628765` |
| Sender wallet | `0x6061873f74E29E686755f9110DB08A8c678f6D52` |
| Input calldata | 884 bytes of UTF-8 JSON (SHA-256 of the input image, the face embedding, the matched URL, etc.) |
| **Screen recording** | **[Google Drive — face-chain-pipeline-e2e.mov](https://drive.google.com/file/d/1xKNrsQEYvfexK2LQQ5oGYyGeTcTODDch/view?usp=sharing)** |

### On-chain payload (decoded from calldata)

```json
{
  "schema": "face-chain-pipeline/v1",
  "run_id": "6ad55a1a-ae0d-4265-becc-67f3db937909",
  "ts_utc": "2026-09-03T20:05:21.503791+00:00",
  "input_image_sha256": "4eba5453a351cb269bf6b4d50835b85bfdb673df78543c4852708f9c91845656",
  "face_embedding_sha256": "14b448b08c4e3940ee267907e231c59a02197bef861cabfbf2a1cc0a17d39f37",
  "face_embedding_dim": 512,
  "face_meta": {
    "det_score": 0.8658279180526733,
    "age": 49,
    "gender": 1,
    "bbox": [117.82, 63.76, 323.08, 335.58]
  },
  "match": {
    "url": "https://www.instagram.com/p/DZtLoP-CFLR/",
    "bucket": "pages_with_matching_images",
    "matched_image_sha256": "279711b7e33701bc8a50cd2175a2374a075a8ec203cbb127979faf45ad16c329"
  },
  "vision_response_summary": {
    "provider": "SerpAPI (Google Lens)",
    "n_pages_with_matching_images": 459,
    "n_full_matching_images": 459,
    "n_visually_similar_images": 459
  },
  "reverse_image_provider": "SerpAPI (Google Lens)"
}
```

Re-verify with `web3.py`:

```python
from web3 import Web3
import json
w3 = Web3(Web3.HTTPProvider("https://ethereum-sepolia-rpc.publicnode.com"))
tx = w3.eth.get_transaction("0xdae079a311874e01046e0b2f9c324bf2bba269b53b70dc797ad9feb357bdbfeb")
print(json.loads(bytes.fromhex(tx.input.hex().removeprefix("0x")).decode("utf-8")))
```

Or open the Etherscan link in a browser → scroll to **Input Data** → **Click to see More**.

---

## What it does

1. **Detect & encode a face** from a single photo using
   [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l`
   pack → 512-d ArcFace embedding, L2-normalized). No external API key
   for this stage; runs entirely on your machine.
2. **Reverse-image search** using
   [SerpAPI Google Lens](https://serpapi.com/google-lens-api) by default
   (no card required, 250 searches/month free — sign up at
   https://serpapi.com/users/sign_up with GitHub or Google). Alternate:
   [Google Cloud Vision `WEB_DETECTION`](https://cloud.google.com/vision/docs/detecting-web)
   if you have a GCP project with billing enabled.
   Both return real public URLs where the image (or visually-similar
   images) appear on the open web, including social-media hosts.
3. **Pick the best matching social-media URL** from the response
   (priority: `pagesWithMatchingImages` → `fullMatchingImages` →
   `partialMatchingImages` → `visuallySimilarImages`, filtered to
   twitter / reddit / instagram / facebook / tiktok / linkedin / youtube /
   tumblr / pinterest / threads).
4. **Anchor a JSON payload** `{input_image_sha256, face_embedding_sha256,
   match_url, matched_image_sha256, ts, ...}` to **Ethereum Sepolia**
   as calldata in a self-send transaction. Anyone can independently
   re-verify by reading the transaction input on Etherscan or any RPC.

Outputs `receipt.json` (full) and `receipt.txt` (human-readable) in
`.receipts/<timestamp>/`.

---

## Why this design

| Decision | Choice | Why |
|---|---|---|
| Face lib | InsightFace `buffalo_l` | 512-d ArcFace, real biometric, no dlib-from-source pain, MIT code, Apple Silicon wheels. |
| Reverse-image | SerpAPI Google Lens (no card, 250/mo free) — alternate: Google Cloud Vision `WEB_DETECTION` | Returns real public URLs across social hosts; SerpAPI needs zero GCP setup, just a GitHub/Google signup. |
| Blockchain | Ethereum Sepolia self-send | Public, re-verifiable by anyone via Etherscan, no node install, faucet available. |
| On-chain format | Raw JSON in `tx.data` | No contract to deploy, re-read via `w3.eth.get_transaction(tx_hash)`, visible on Etherscan. |
| Demo subject | Cristiano Ronaldo (Wikimedia Commons, CC-licensed) | Public figure guaranteed to surface in `WEB_DETECTION` results. The pipeline is face-agnostic — see "Limitations" below. |

---

## Setup

Tested on macOS Apple Silicon, Python 3.11.

```bash
git clone https://github.com/<you>/face-chain-pipeline
cd face-chain-pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### One-time secrets

1. **Reverse-image search (SerpAPI — recommended, no card, no GCP)**
   - Sign up at https://serpapi.com/users/sign_up with GitHub or Google
   - No payment method required; free tier is 250 searches/month and never auto-converts
   - Dashboard at https://serpapi.com/manage-api-key shows your key
2. **(Alternate) Google Cloud Vision**
   - Create a GCP project: https://console.cloud.google.com
   - Enable the **Cloud Vision API**
   - Create an API key under **APIs & Services → Credentials**
   - **A credit card is required** and a minimum prepayment is enforced
     in some regions. Set a budget alert for $1 to be safe
3. **Sepolia RPC + wallet**
   - Free Alchemy key: https://alchemy.com → create app → Sepolia
   - Create a throwaway wallet, export its private key
   - Fund it: https://cloud.google.com/application/web3/faucet/ethereum/sepolia
     (Google faucet — no mainnet ETH required, no card)

Then:
```bash
cp .env.example .env
# edit .env: fill in SERPAPI_KEY (or GOOGLE_VISION_API_KEY), SEPOLIA_RPC_URL, SEPOLIA_PRIVATE_KEY
```

---

## Run

```bash
python pipeline.py                       # default: downloads Ronaldo from Wikimedia
python pipeline.py path/to/your.jpg     # any face photo
```

The first run downloads:
- `buffalo_l` face-analysis pack (~326 MB) to `~/.insightface/models/`
- The demo image (or your local file is used directly)

The full run takes ~60–90 seconds after the one-time model download.

---

## Re-verify the on-chain record

The pipeline prints the Etherscan link. Independently:

```python
from web3 import Web3
w3 = Web3(Web3.HTTPProvider("<your_sepolia_rpc>"))
tx = w3.eth.get_transaction("<tx_hash>")
import json
print(json.loads(bytes.fromhex(tx.input.hex().removeprefix("0x")).decode()))
```

Or open `<etherscan_url>` in a browser and click **Click to see More** → **Input Data**.

---

## Known limitations (read me, this matters)

- **On-chain payload is a *claim*, not a proof.** The Sepolia transaction
  attests that at block N, the pipeline asserted "this face matches this
  URL." It does not cryptographically prove the social-media URL actually
  contains that face. Re-running the pipeline on the same input may
  produce a different match URL — SerpAPI Google Lens returns an
  ordered list of visually-similar pages and the exact top match
  depends on the day. The hash of the matched image and the embedding
  hash are both anchored so a third party can re-derive and compare
  — but trust in the *face ↔ URL mapping* ultimately rests on the
  reverse-image provider (SerpAPI/Google Lens or Google Cloud Vision),
  not on the chain.
- **Some SerpAPI result URLs are wrapped in Google `goto?url=CAES...`
  redirects.** These resolve client-side via JavaScript in a real
  browser, so Python's `requests` library can't follow them. The
  pipeline handles this by ranking direct social-media URLs
  (`instagram.com/p/...`, `reddit.com/r/.../comments/...`) higher
  than wrapped ones, and falling back to title/source matching when
  no direct URL is available. The verified run picked a *direct*
  Instagram URL on the first try.
- **Testnet permanence is not archival.** Sepolia history is maintained
  by client teams; there's no SLA. Fine for a demo, don't pitch as
  permanent notarization.
- **Single-writer trust.** The pipeline's wallet is the only writer. The
  chain prevents *backdating* and *hiding*; it does not prevent the same
  writer from publishing a contradictory claim later. For "even the
  writer can't forge," use [OpenTimestamps](https://opentimestamps.org)
  (Bitcoin-anchored) — at the cost of hours-of-latency for verification.
- **`buffalo_l` model license** is for non-commercial research only.
  Swap to `buffalo_s` (or a fully permissive model) for commercial use.
- **GCP requires a card on file** even for the free tier. Set a budget
  alert. Each `WEB_DETECTION` call is ~10 units; you can run ~100 demos
  per month free. **If you don't want to deal with GCP at all, the
  default reverse-image provider is SerpAPI Google Lens (no card,
  250/month free)** — just sign up with GitHub or Google.
- **InsightFace `buffalo_l` is 326 MB.** First-run download is slow.
- **Demo subject = Cristiano Ronaldo** because a public figure is the
  only way to *guarantee* a real social-media match in a screen
  recording. The pipeline has no hardcoded search results and works on
  any face — but expect zero or weak matches on a private individual's
  photo, which is a feature of the open web, not a bug.

---

## Repo layout

```
face-chain-pipeline/
├── pipeline.py            # the whole thing, ~280 lines
├── requirements.txt
├── .env.example           # copy to .env and fill in
├── .gitignore
└── README.md
```

No website. No frontend. No hosting. The pipeline is the deliverable.
