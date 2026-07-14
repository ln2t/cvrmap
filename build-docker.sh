#!/bin/bash

# CVRmap Docker Build Script
set -e

# Get version from cvrmap/__init__.py (single source of truth)
VERSION=$(python -c "
import re
with open('cvrmap/__init__.py', 'r') as f:
    content = f.read()
version_match = re.search(r'__version__\s*=\s*[\"\'](.*?)[\"\']', content)
if version_match:
    print(version_match.group(1))
else:
    print('0.1.0')
" 2>/dev/null || echo "0.1.0")

# Default values
IMAGE_NAME="arovai/cvrmap"
DOCKERFILE="Dockerfile"
BUILD_ARGS=""
PUSH=false
LATEST=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -f|--file)
            DOCKERFILE="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --latest)
            LATEST=true
            shift
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        -h|--help)
            echo "CVRmap Docker Build Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --version VERSION   Set version tag (default: auto-detected from cvrmap/__init__.py)"
            echo "  -i, --image IMAGE       Set image name (default: arovai/cvrmap)"
            echo "  -f, --file DOCKERFILE   Set Dockerfile path (default: Dockerfile)"
            echo "  --push                  Push image to registry after build"
            echo "  --latest                Also tag as 'latest'"
            echo "  --build-arg ARG=VALUE   Pass build argument to Docker"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Build with auto-detected version"
            echo "  $0 -v 1.0.0 --push --latest         # Build v1.0.0 and push with latest tag"
            echo "  $0 --build-arg PYTHON_VERSION=3.9    # Build with custom Python version"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}         CVRmap Docker Build${NC}"
echo -e "${BLUE}===========================================${NC}"
echo -e "${YELLOW}Image:${NC} $IMAGE_NAME"
echo -e "${YELLOW}Version:${NC} $VERSION"
echo -e "${YELLOW}Dockerfile:${NC} $DOCKERFILE"
echo -e "${BLUE}===========================================${NC}"

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    echo -e "${RED}Error: Dockerfile '$DOCKERFILE' not found${NC}"
    exit 1
fi

# Build the image
echo -e "${GREEN}Building Docker image...${NC}"
docker build \
    $BUILD_ARGS \
    --build-arg CVRMAP_VERSION="$VERSION" \
    -t "$IMAGE_NAME:$VERSION" \
    -f "$DOCKERFILE" \
    .

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Successfully built $IMAGE_NAME:$VERSION${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Tag as latest if requested
if [[ "$LATEST" == "true" ]]; then
    echo -e "${GREEN}Tagging as latest...${NC}"
    docker tag "$IMAGE_NAME:$VERSION" "$IMAGE_NAME:latest"
    echo -e "${GREEN}✓ Tagged as $IMAGE_NAME:latest${NC}"
fi

# Push if requested
if [[ "$PUSH" == "true" ]]; then
    echo -e "${GREEN}Pushing to registry...${NC}"
    docker push "$IMAGE_NAME:$VERSION"
    
    if [[ "$LATEST" == "true" ]]; then
        docker push "$IMAGE_NAME:latest"
    fi
    
    echo -e "${GREEN}✓ Successfully pushed to registry${NC}"
fi

echo -e "${BLUE}===========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""
echo "Usage examples:"
echo -e "  ${YELLOW}docker run --rm $IMAGE_NAME:$VERSION --help${NC}"
echo -e "  ${YELLOW}docker run --rm -v /path/to/data:/data/input -v /path/to/output:/data/output $IMAGE_NAME:$VERSION /data/input /data/output participant --task gas${NC}"
echo ""
echo "With Docker Compose:"
echo -e "  ${YELLOW}docker-compose run cvrmap /data/input /data/output participant --task gas${NC}"
