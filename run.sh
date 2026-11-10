#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x build/flash_attention_tests ]; then
    echo "Executable not found; build the project first with ./build.sh make." >&2
    exit 1
fi

if [ "${1:-}" == "list" ]; then
    if [ "$#" -ne 1 ]; then
        echo "Usage: ./run.sh [list|<test-name-or-filter> ...]" >&2
        exit 1
    fi
    exec ./build/flash_attention_tests --gtest_list_tests
fi

if [ "$#" -eq 0 ]; then
    exec ./build/flash_attention_tests
fi

filter="$1"
shift
for test_name in "$@"; do
    filter+=":$test_name"
done

matched_tests=$(./build/flash_attention_tests --gtest_list_tests \
                    "--gtest_filter=$filter" | grep -c '^  ' || true)
if [ "$matched_tests" -eq 0 ]; then
    echo "No tests match filter: $filter" >&2
    echo "Use './run.sh list' to see the full test names." >&2
    exit 1
fi

exec ./build/flash_attention_tests "--gtest_filter=$filter"
