"""Compare the deployment contracts of two exported policies, e.g. a jump policy against the walking policy.

Each policy is a ``policy.onnx`` path, or a Hugging Face model repo (``org/name``) holding one. Exits with status 1
when the contracts differ.

.. code-block:: bash

    ./cyclotron.sh --compare-contract logs/rsl_rl/asimov1_jump/<run>/exported/policy.onnx \\
        Menlo/asimov1-locomotion-26-09-24
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "cyclotron"))

from cyclotron.deploy_contract import CONTRACT_KEYS, contract_differences, read_contract  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("first", type=str, help="policy.onnx, or a Hugging Face repo with one.")
parser.add_argument("second", type=str, help="policy.onnx, or a Hugging Face repo with one.")


def resolve(policy: str) -> str:
    if os.path.isfile(policy):
        return policy
    from huggingface_hub import hf_hub_download

    return hf_hub_download(policy, "policy.onnx")


def main() -> None:
    args = parser.parse_args()
    first, second = (read_contract(resolve(policy)) for policy in (args.first, args.second))
    if not any(key in first for key in CONTRACT_KEYS) or not any(key in second for key in CONTRACT_KEYS):
        sys.exit("[ERROR] A policy has no deploy metadata: export it with ./cyclotron.sh --export.")
    differences = contract_differences(first, second)
    if differences:
        print(f"[ERROR] The deployment contracts differ on {len(differences)} keys:")
        for line in differences:
            print(f"  {line}")
        sys.exit(1)
    print(f"[INFO] Same deployment contract ({', '.join(key for key in CONTRACT_KEYS if key in first)}).")


if __name__ == "__main__":
    main()
