#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

rsync -avz -e 'ssh -p 2222' \
      --exclude-from='.rsyncignore' \
      zhuwenjie@100.93.80.22:~/work/flash_attention/ ./
