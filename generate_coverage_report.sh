#!/bin/bash
#
# Code Coverage Report Generator for Zertz3D
#
# Generates comprehensive code coverage reports using pytest-cov.
# Produces both terminal output and HTML reports.
#
# Usage:
#   ./generate_coverage_report.sh [OPTIONS]
#
# Options:
#   --html-only    Generate only HTML report (skip terminal output)
#   --open         Open HTML report in browser after generation
#   --game         Report coverage only for game/ directory
#   --learner      Report coverage only for learner/ directory
#   --renderer     Report coverage only for renderer/ directory
#   --fast         Skip slow tests (marked with @pytest.mark.slow)
#   --help         Show this help message
#

set -e  # Exit on error

# Default options
HTML_ONLY=false
OPEN_BROWSER=false
COVERAGE_PATH="."
SKIP_SLOW=false
EXTRA_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --html-only)
            HTML_ONLY=true
            shift
            ;;
        --open)
            OPEN_BROWSER=true
            shift
            ;;
        --game)
            COVERAGE_PATH="game"
            shift
            ;;
        --learner)
            COVERAGE_PATH="learner"
            shift
            ;;
        --renderer)
            COVERAGE_PATH="renderer"
            shift
            ;;
        --fast)
            SKIP_SLOW=true
            EXTRA_ARGS="$EXTRA_ARGS -m 'not slow'"
            shift
            ;;
        --help)
            grep '^#' "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Zertz3D Code Coverage Report${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if pytest-cov is installed
if ! python3 -m pytest --co -q --cov 2>/dev/null | grep -q "coverage"; then
    echo -e "${YELLOW}Warning: pytest-cov not found. Installing...${NC}"
    uv pip install pytest-cov
fi

# Create htmlcov directory if it doesn't exist
mkdir -p htmlcov

# Build coverage command
COV_CMD="python3 -m pytest tests/"

# Add coverage options
if [ "$HTML_ONLY" = true ]; then
    COV_CMD="$COV_CMD --cov=$COVERAGE_PATH --cov-report=html --no-cov-on-screen"
else
    COV_CMD="$COV_CMD --cov=$COVERAGE_PATH --cov-report=term-missing --cov-report=html"
fi

# Add exclusions
COV_CMD="$COV_CMD --cov-config=.coveragerc"

# Add extra arguments
if [ -n "$EXTRA_ARGS" ]; then
    COV_CMD="$COV_CMD $EXTRA_ARGS"
fi

echo -e "${GREEN}Running coverage analysis...${NC}"
echo -e "Coverage path: ${YELLOW}$COVERAGE_PATH${NC}"
if [ "$SKIP_SLOW" = true ]; then
    echo -e "Skipping slow tests"
fi
echo ""

# Run coverage
eval $COV_CMD

# Check if coverage report was generated
if [ ! -f "htmlcov/index.html" ]; then
    echo -e "${YELLOW}Warning: HTML coverage report was not generated${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Coverage report generated successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "HTML report: ${BLUE}htmlcov/index.html${NC}"
echo ""

# Open in browser if requested
if [ "$OPEN_BROWSER" = true ]; then
    echo -e "${GREEN}Opening coverage report in browser...${NC}"
    if command -v open &> /dev/null; then
        # macOS
        open htmlcov/index.html
    elif command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open htmlcov/index.html
    elif command -v start &> /dev/null; then
        # Windows
        start htmlcov/index.html
    else
        echo -e "${YELLOW}Could not detect browser opener. Please open htmlcov/index.html manually.${NC}"
    fi
fi

echo -e "${BLUE}========================================${NC}"
echo ""
echo "To view the report, run:"
echo "  open htmlcov/index.html"
echo ""
echo "Or generate a quick terminal report:"
echo "  python3 -m pytest tests/ --cov=$COVERAGE_PATH --cov-report=term-missing"
echo ""