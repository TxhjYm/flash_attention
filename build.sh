#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [ "${1:-}" == "clean" ]; then
    rm -rf build
    echo "Clean done."
elif [ "${1:-}" == "make" ]; then
    cmake -S . -B build
    cmake --build build --parallel
else
    echo "Usage: ./build.sh [clean|make]"
    exit 1
fi
