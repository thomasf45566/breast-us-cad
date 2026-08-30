"""Verify the live Hugging Face Space against the local frozen pipeline.

Research prototype — not for clinical use. Not a medical device.

Measures cold-start (sleeping/restarted -> first successful prediction) and
warm latency, runs the four bundled examples through the live Space, and
compares the full-precision probabilities (embedded in the result card as an
HTML comment) with reports/app_example_probs.json recorded locally.

Usage: python scripts/verify_space.py [--cold]
       --cold: restart the Space first to measure a true cold start
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from gradio_client import Client, handle_file
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "reports" / "app_example_probs.json").read_text())
EXAMPLES = sorted((ROOT / "app" / "examples").glob("*.png"))
COLD_TIMEOUT_S = 1800


def predict(client: Client, image_path: Path) -> tuple[str, str, float]:
    t0 = time.perf_counter()
    result = client.predict(handle_file(str(image_path)), api_name="/predict")
    dt = time.perf_counter() - t0
    card = result[2]
    m = re.search(r"<!-- raw=([\d.e-]+) cal=([\d.e-]+) -->", card)
    if not m:
        raise RuntimeError(f"No full-precision comment in result card:\n{card[-500:]}")
    return m.group(1), m.group(2), dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold", action="store_true", help="restart Space first")
    parser.add_argument("--space", help="override <user>/breast-us-cad")
    args = parser.parse_args()

    api = HfApi()
    space = args.space or f"{api.whoami()['name']}/breast-us-cad"
    print(f"space: {space}")

    if args.cold:
        print("restarting Space for a cold-start measurement ...")
        api.restart_space(space)
        time.sleep(2)

    t0 = time.perf_counter()
    client = None
    while time.perf_counter() - t0 < COLD_TIMEOUT_S:
        try:
            client = Client(space, verbose=False)
            raw, cal, _ = predict(client, EXAMPLES[0])
            break
        except Exception as e:
            print(f"  waiting ({time.perf_counter() - t0:.0f}s): {type(e).__name__}")
            client = None
            time.sleep(10)
    if client is None:
        raise SystemExit("Space did not come up within the timeout")
    cold = time.perf_counter() - t0
    label = "cold start (restart -> first prediction)" if args.cold else "first prediction (already warm?)"
    print(f"{label}: {cold:.1f} s")

    print("\nfour-example comparison (live vs local, full precision):")
    all_match = True
    warm_times = []
    for p in EXAMPLES:
        raw, cal, dt = predict(client, p)
        exp = EXPECTED[p.name]
        match = raw == exp["raw"] and cal == exp["cal"]
        all_match &= match
        warm_times.append(dt)
        print(f"  {p.name}: raw={raw} cal={cal} "
              f"{'EXACT MATCH' if match else 'MISMATCH (expected raw=%s cal=%s)' % (exp['raw'], exp['cal'])} "
              f"[{dt:.2f} s]")
    for _ in range(2):  # a couple extra timing runs on one image
        warm_times.append(predict(client, EXAMPLES[0])[2])
    print(f"\nwarm latency: median {statistics.median(warm_times):.2f} s "
          f"(min {min(warm_times):.2f} / max {max(warm_times):.2f}, n={len(warm_times)})")
    print("RESULT:", "ALL FOUR PROBABILITIES MATCH LOCAL EXACTLY" if all_match
          else "PROBABILITY MISMATCH — investigate before recording")


if __name__ == "__main__":
    main()
