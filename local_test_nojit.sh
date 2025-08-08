#!/bin/bash
# local_test.sh

echo "🔍 Testing FinMLKit CI locally..."

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

# Set PYTHONPATH and DISABLE JIT for testing
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export NUMBA_DISABLE_JIT=1
echo "⚠️  JIT compilation disabled for testing"

# Run tests with coverage from project root
echo "🧪 Running tests..."
pytest tests/ --cov=finmlkit --cov-report=xml --cov-report=term -v

# Check if coverage.xml was created
if [ -f "coverage.xml" ]; then
    echo "✅ Coverage report generated successfully"
else
    echo "❌ Coverage report not found"
    deactivate
    rm -rf temp_test_env
    exit 1
fi

# Deactivate and remove temp environment
echo "🧹 Cleaning up environment..."
deactivate
rm -rf temp_test_env

echo "🎉 Local CI test completed successfully!"