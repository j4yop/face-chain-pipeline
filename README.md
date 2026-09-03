# face-chain-pipeline

End-to-end pipeline: **face scan → reverse-image search → blockchain anchor**.

A single `python pipeline.py` run does all four stages and writes a verifiable
on-chain receipt on Ethereum Sepolia.

---

## What it does

1. **Detect & encode a face** from a single photo using
   [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l`
   pack → 512-d ArcFace embedding, L2-normalized). No external API key
   for this stage; runs entirely on your machine.
2. **Reverse-image search** using
   [Google Cloud Vision `WEB_DETECTION`](https://cloud.google.com/vision/docs/detecting-web).
   Returns real public URLs where the image (or visually-similar images)
   appear on the open web, including social-media hosts.
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
| Reverse-image | Google Cloud Vision `WEB_DETECTION` | Returns real public URLs across social hosts, 1k units/month free, stable Python client, no scraping. |
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

1. **Google Cloud Vision**
   - Create a GCP project: https://console.cloud.google.com
   - Enable the **Cloud Vision API**
   - Create a service account, download its JSON key
   - **Card on file is required** but you stay inside the 1,000-units/month
     free tier and the $300 GCP trial credit, so nothing is charged
   - Set a budget alert for $1 in Billing to be safe
2. **Sepolia RPC + wallet**
   - Free Alchemy key: https://alchemy.com → create app → Sepolia
   - Create a throwaway wallet, export its private key
   - Fund it: https://faucet.quicknode.com/ethereum/sepolia (≈0.1 ETH free)

Then:
```bash
cp .env.example .env
# edit .env: fill in GOOGLE_APPLICATION_CREDENTIALS, SEPOLIA_RPC_URL, SEPOLIA_PRIVATE_KEY
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
  produce a different `pagesWithMatchingImages` ordering. The hash of
  the matched image and the embedding hash are both anchored so a third
  party can re-derive and compare — but trust in the *face ↔ URL mapping*
  ultimately rests on Google Vision's `WEB_DETECTION`, not on the chain.
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
  per month free.
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
