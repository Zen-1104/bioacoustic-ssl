#!/usr/bin/env bash
set -euo pipefail

# Run on a network-enabled machine/login node, not an offline compute node.
# Zenodo publishes this MD5 for Cnn14_mAP=0.431.pth. The script verifies it
# before promoting the temporary download, then reports SHA-256 for records.
readonly expected_md5="541141fa2ee191a88f24a3219fff024e"
readonly expected_sha256="0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"
out="${1:-data/pretrained/panns/Cnn14_mAP=0.431.pth}"
mkdir -p "$(dirname "$out")"
tmp="${out}.partial"
trap 'rm -f "$tmp"' EXIT
curl --fail --location --retry 3 --output "$tmp" \
  'https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
test -s "$tmp"
actual_md5="$(python - "$tmp" <<'PYMD5'
import hashlib, sys
h = hashlib.md5()
with open(sys.argv[1], "rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
print(h.hexdigest())
PYMD5
)"
if [[ "$actual_md5" != "$expected_md5" ]]; then
  echo "ERROR: PANNs checkpoint MD5 mismatch; expected $expected_md5, got $actual_md5" >&2
  exit 3
fi
if command -v sha256sum >/dev/null 2>&1; then
  actual_sha256="$(sha256sum "$tmp" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  actual_sha256="$(shasum -a 256 "$tmp" | awk '{print $1}')"
else
  printf 'ERROR: no SHA-256 utility is available; refusing to accept an unverified checkpoint.\n' >&2
  exit 2
fi
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "ERROR: PANNs checkpoint SHA-256 mismatch; expected $expected_sha256, got $actual_sha256" >&2
  exit 4
fi
mv "$tmp" "$out"
trap - EXIT
printf '%s  %s\n' "$actual_sha256" "$out"
printf 'Official PANNs checkpoint ready: %s (MD5 and SHA-256 verified)\n' "$out"
