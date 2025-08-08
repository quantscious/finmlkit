#!/bin/bash
# ./local_test.sh

echo "🔍 Running FinMLKit tests individually with coverage collection..."

# Ensure we're in the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Create and activate temp virtual environment
echo "🆕 Creating temporary virtual environment..."
python -m venv temp_test_env
source temp_test_env/bin/activate

# Install in development mode
echo "📦 Installing package..."
pip install --upgrade pip
pip install -e .[dev]

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Clean up any existing coverage files
rm -f .coverage*
rm -f coverage.xml

# Initialize counters
total_files=0
passed_files=0
failed_files=0
failed_file_list=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🧪 Running tests individually with coverage collection..."
echo "======================================================"

# Function to run tests for a directory
run_tests_in_directory() {
    local test_dir=$1
    local dir_name=$(basename "$test_dir")

    if [ ! -d "$test_dir" ]; then
        echo "⚠️  Directory $test_dir does not exist, skipping..."
        return
    fi

    echo ""
    echo "📁 Testing directory: $dir_name"
    echo "--------------------------------"

    # Find all test files in the directory
    for test_file in "$test_dir"/test_*.py; do
        if [ -f "$test_file" ]; then
            total_files=$((total_files + 1))
            local filename=$(basename "$test_file")

            echo -n "  🔬 $filename... "

            # Run the individual test file WITH coverage collection
            if pytest "$test_file" --cov=finmlkit --cov-append --cov-report= -v --tb=short > /dev/null 2>&1; then
                echo -e "${GREEN}PASS${NC}"
                passed_files=$((passed_files + 1))
            else
                echo -e "${RED}FAIL${NC}"
                failed_files=$((failed_files + 1))
                failed_file_list+=("$test_file")
            fi
        fi
    done
}

# Run tests for each directory
run_tests_in_directory "tests/bars"
run_tests_in_directory "tests/features"
run_tests_in_directory "tests/labels"
run_tests_in_directory "tests/sampling"
run_tests_in_directory "tests/structural_breaks"
run_tests_in_directory "tests/integration"

echo ""
echo "================================"
echo "📊 Test Results Summary"
echo "================================"
echo "Total test files: $total_files"
echo -e "Passed: ${GREEN}$passed_files${NC}"
echo -e "Failed: ${RED}$failed_files${NC}"

# Show failed files if any
if [ $failed_files -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Failed test files:${NC}"
    for failed_file in "${failed_file_list[@]}"; do
        echo "  - $failed_file"
    done
    echo ""
    echo "🔍 To debug a specific failure, run:"
    echo "   pytest ${failed_file_list[0]} -v"
fi

# Generate final coverage reports from collected data
echo ""
echo "📈 Generating coverage reports from collected data..."

if [ -f ".coverage" ]; then
    # Generate XML and terminal reports from the collected coverage data
    python -m coverage xml
    python -m coverage report --show-missing

    if [ -f "coverage.xml" ]; then
        echo "✅ Coverage report generated successfully"

        # Show coverage summary
        echo ""
        echo "📊 Coverage Summary:"
        python -m coverage report --format=total
    else
        echo "⚠️  XML coverage report not found"
    fi
else
    echo "⚠️  No coverage data found"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up environment..."
deactivate
rm -rf temp_test_env

echo ""
if [ $failed_files -eq 0 ]; then
    echo "🎉 All tests passed successfully with coverage collected!"
    exit 0
else
    echo "💥 Some tests failed. Check the failed files listed above."
    exit 1
fi