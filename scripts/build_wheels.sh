#!/bin/bash
# Build Python wheels for multiple platforms
#
# This script provides options for building wheels locally:
# 1. Native build (current platform only)
# 2. Cross-compilation (requires setup)
# 3. Docker-based manylinux builds (Linux only)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Zertz3D Wheel Builder ===${NC}\n"

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo -e "${RED}Error: maturin not found${NC}"
    echo "Install with: pip install maturin"
    exit 1
fi

# Parse command line arguments
BUILD_TYPE="${1:-native}"

case "$BUILD_TYPE" in
    native)
        echo -e "${GREEN}Building wheel for current platform...${NC}"
        maturin build --release --out dist
        echo -e "\n${GREEN}✓ Native wheel built successfully${NC}"
        echo -e "Output: ${YELLOW}dist/*.whl${NC}"
        ;;

    manylinux)
        if [[ "$OSTYPE" != "linux-gnu"* ]]; then
            echo -e "${RED}Error: manylinux builds only work on Linux${NC}"
            echo "Use Docker on macOS/Windows (see below)"
            exit 1
        fi

        echo -e "${GREEN}Building manylinux wheels...${NC}"
        docker run --rm -v "$(pwd)":/io \
            quay.io/pypa/manylinux_2_28_x86_64 \
            bash -c "cd /io && \
                     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
                     source \$HOME/.cargo/env && \
                     /opt/python/cp312-cp312/bin/pip install maturin && \
                     /opt/python/cp312-cp312/bin/maturin build --release --out dist"

        echo -e "\n${GREEN}✓ Manylinux wheels built successfully${NC}"
        echo -e "Output: ${YELLOW}dist/*.whl${NC}"
        ;;

    manylinux-docker)
        echo -e "${GREEN}Building manylinux wheels using Docker...${NC}"

        if ! command -v docker &> /dev/null; then
            echo -e "${RED}Error: Docker not found${NC}"
            echo "Install Docker Desktop from https://www.docker.com/products/docker-desktop"
            exit 1
        fi

        # Temporarily remove Cargo.lock (will be regenerated inside container with correct version)
        CARGO_LOCK_BACKUP=""
        if [ -f "rust/Cargo.lock" ]; then
            echo -e "${YELLOW}Backing up Cargo.lock (will be regenerated for Docker build)${NC}"
            CARGO_LOCK_BACKUP="$(cat rust/Cargo.lock)"
            rm rust/Cargo.lock
        fi

        # Use PyO3 maturin-action docker image with platform specification
        docker run --platform linux/amd64 --rm -v "$(pwd)":/io \
            ghcr.io/pyo3/maturin:latest \
            build --release --out /io/dist

        # Restore Cargo.lock
        if [ -n "$CARGO_LOCK_BACKUP" ]; then
            echo "$CARGO_LOCK_BACKUP" > rust/Cargo.lock
        fi

        echo -e "\n${GREEN}✓ Manylinux wheels built successfully${NC}"
        echo -e "Output: ${YELLOW}dist/*.whl${NC}"
        ;;

    all-docker)
        echo -e "${GREEN}Building wheels for all platforms using Docker...${NC}"
        echo -e "${YELLOW}Note: This requires Docker and will take several minutes${NC}\n"

        if ! command -v docker &> /dev/null; then
            echo -e "${RED}Error: Docker not found${NC}"
            echo "Install Docker Desktop from https://www.docker.com/products/docker-desktop"
            exit 1
        fi

        # Create dist directory
        mkdir -p dist

        # Temporarily remove Cargo.lock (will be regenerated inside container with correct version)
        CARGO_LOCK_BACKUP=""
        if [ -f "rust/Cargo.lock" ]; then
            echo -e "${YELLOW}Backing up Cargo.lock (will be regenerated for Docker build)${NC}"
            CARGO_LOCK_BACKUP="$(cat rust/Cargo.lock)"
            rm rust/Cargo.lock
        fi

        echo -e "\n${GREEN}Building Linux (manylinux) wheels...${NC}"
        docker run --platform linux/amd64 --rm -v "$(pwd)":/io \
            ghcr.io/pyo3/maturin:latest \
            build --release --out /io/dist

        # Note: Cross-compiling to Windows from Linux has limitations
        # For production Windows builds, build natively on Windows
        echo -e "\n${YELLOW}Note: Skipping Windows build (requires native Windows or complex cross-compilation)${NC}"
        echo -e "${YELLOW}For Windows wheels, either:${NC}"
        echo -e "${YELLOW}  1. Build natively on Windows using: maturin build --release${NC}"
        echo -e "${YELLOW}  2. Use GitHub Actions (rename .github/workflows/build-wheels.yml.disabled to .yml)${NC}"

        # Restore Cargo.lock
        if [ -n "$CARGO_LOCK_BACKUP" ]; then
            echo "$CARGO_LOCK_BACKUP" > rust/Cargo.lock
        fi

        echo -e "\n${GREEN}✓ Linux wheels built successfully${NC}"
        echo -e "Output: ${YELLOW}dist/*.whl${NC}"
        ;;

    clean)
        echo -e "${YELLOW}Cleaning build artifacts...${NC}"
        rm -rf dist/
        rm -rf target/
        rm -rf rust/target/
        echo -e "${GREEN}✓ Clean complete${NC}"
        ;;

    *)
        echo -e "${RED}Unknown build type: $BUILD_TYPE${NC}\n"
        echo "Usage: $0 [native|manylinux|manylinux-docker|all-docker|clean]"
        echo ""
        echo "Options:"
        echo "  native           - Build wheel for current platform (default)"
        echo "  manylinux        - Build manylinux wheels (Linux only)"
        echo "  manylinux-docker - Build manylinux wheels using Docker (any platform)"
        echo "  all-docker       - Build wheels for Linux and Windows using Docker"
        echo "  clean            - Remove build artifacts"
        echo ""
        echo "Examples:"
        echo "  $0                    # Build for current platform"
        echo "  $0 native             # Same as above"
        echo "  $0 manylinux-docker   # Build Linux wheels using Docker"
        echo "  $0 all-docker         # Build all platforms using Docker"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Build complete!${NC}"
echo "Install with: pip install dist/*.whl"