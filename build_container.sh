#!/bin/bash
#
# Build script for CVRmap Apptainer container
#
# Usage:
#   ./build_container.sh [--sandbox] [--fakeroot]
#
# Options:
#   --sandbox   Build as a writable sandbox directory (for development/debugging)
#   --fakeroot  Build without root privileges using fakeroot
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Extract version from cvrmap/__init__.py (single source of truth)
VERSION=$(grep -oP "__version__\s*=\s*\"\K[^\"]*" "${SCRIPT_DIR}/cvrmap/__init__.py")
CONTAINER_NAME="cvrmap_${VERSION}.sif"

# Parse arguments
SANDBOX=false
FAKEROOT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --sandbox)
            SANDBOX=true
            shift
            ;;
        --fakeroot)
            FAKEROOT="--fakeroot"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Building CVRmap Apptainer Container"
echo "Version: ${VERSION}"
echo "========================================"

cd "${SCRIPT_DIR}"

if [ "$SANDBOX" = true ]; then
    echo "Building as sandbox..."
    CONTAINER_NAME="cvrmap_${VERSION}_sandbox"
    apptainer build ${FAKEROOT} --sandbox "${CONTAINER_NAME}" Apptainer
    echo ""
    echo "Sandbox created: ${CONTAINER_NAME}/"
    echo "To enter the sandbox: apptainer shell --writable ${CONTAINER_NAME}/"
else
    echo "Building SIF container..."
    apptainer build ${FAKEROOT} "${CONTAINER_NAME}" Apptainer
    echo ""
    echo "Container built: ${CONTAINER_NAME}"
fi

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo ""
echo "To run the container:"
echo "  apptainer run ${CONTAINER_NAME} --help"
echo ""
echo "Example usage:"
echo "  apptainer run ${CONTAINER_NAME} /data/bids /data/derivatives participant \\"
echo "      --derivatives fmriprep=/data/derivatives/fmriprep \\"
echo "      --task gas --participant_label 001"
