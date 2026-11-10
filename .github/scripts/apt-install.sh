#!/usr/bin/env bash
# Install only the apt packages the runner image lacks. GitHub's Ubuntu images already ship
# most toolchains, and a slow mirror once stalled a job for over ten minutes in apt-get.
set -euo pipefail
missing=()
for pkg in "$@"; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if [ "${#missing[@]}" -eq 0 ]; then
  echo "already installed: $*"
  exit 0
fi
echo "installing: ${missing[*]}"
sudo timeout 600 apt-get update -qq
sudo timeout 600 apt-get install -y -qq "${missing[@]}"
