#!/usr/bin/env bash
# Build the checksummed archive tarballs for the NPE convergence paper data.
#
# Usage: bash scripts/build_archive_tarballs.sh /Volumes/YourDrive/npe_archive
#
# Run from the repo root. DEST needs about 10 GiB free. The archive subset is
# the 18 paths in docs/coauthor_report_2026_05_31/zenodo_filelist.txt (the
# 2026-05-31 Zenodo staging plan) plus the June 2026 overnight results and a
# paper-source snapshot. The same tarballs can later be uploaded to Zenodo.
set -euo pipefail

DEST="${1:?usage: build_archive_tarballs.sh <destination dir>}"
STAMP=20260709
FILELIST=docs/coauthor_report_2026_05_31/zenodo_filelist.txt

mkdir -p "$DEST"

echo "[1/3] results subset (~7.3 GiB input)"
tar -czf "$DEST/npe_convergence_results_${STAMP}.tar.gz" \
    -T "$FILELIST" \
    res/overnight_20260601 \
    res/gnk_model_control_n5000 \
    res/stereological_blackjax_smc_abc

echo "[2/3] paper source and figure snapshot"
tar -czf "$DEST/npe_convergence_paper_snapshot_${STAMP}.tar.gz" \
    paper.tex notebooks/plots \
    notebooks/coauthor_report_2026_05_31 \
    docs/coauthor_report_2026_05_31

echo "[3/3] checksums"
(cd "$DEST" && shasum -a 256 npe_convergence_*_${STAMP}.tar.gz > "CHECKSUMS_${STAMP}.sha256")

ls -lh "$DEST"
echo "Done. Verify later with: (cd $DEST && shasum -a 256 -c CHECKSUMS_${STAMP}.sha256)"
