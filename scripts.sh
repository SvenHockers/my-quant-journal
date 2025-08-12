#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

create_venv() {
    print_header "Setting up Virtual Environment"
    
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_status "Virtual environment created successfully!"
    else
        print_status "Virtual environment already exists."
    fi
}

activate_venv() {
    if [ -d "venv" ]; then
        print_status "Activating virtual environment..."
        source venv/bin/activate
        print_status "Virtual environment activated!"
    else
        print_error "Virtual environment not found. Please run setup-dev first."
        exit 1
    fi
}

install_deps() {
    print_header "Installing Dependencies"
    create_venv
    activate_venv
    print_status "Upgrading pip..."
    pip install --upgrade pip

    print_status "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    
    print_status "Dependencies installed successfully!"
    print_warning "Remember to activate the virtual environment with: source venv/bin/activate"
}

run_tests() {
    print_header "Running Tests"
    python -m pytest tests/ -v
}

clean_files() {
    print_header "Cleaning Generated Files"
    rm -rf __pycache__/
    rm -rf src/__pycache__/
    rm -rf src/*/__pycache__/
    rm -rf tests/__pycache__/
    rm -rf results/
    rm -rf analysis/
    rm -rf data/cache/
    rm -rf logs/
    rm -rf .pytest_cache/
    rm -rf .coverage
    rm -rf htmlcov/
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete
    find . -name "*.pyd" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    print_status "Cleanup completed!"
}

clean_venv() {
    print_header "Cleaning Virtual Environment"
    if [ -d "venv" ]; then
        print_warning "This will remove the virtual environment. Are you sure? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf venv/
            print_status "Virtual environment removed!"
        else
            print_status "Virtual environment cleanup cancelled."
        fi
    else
        print_status "No virtual environment found to clean."
    fi
}

run_example() {
    print_header "Running SMA Crossover Example"
    python examples/run_sma_backtest.py
}

run_lint() {
    print_header "Running Linting Checks"
    flake8 src/ tests/ examples/
    mypy src/
}

format_code() {
    print_header "Formatting Code with Black"
    black src/ tests/ examples/
    print_status "Code formatting completed!"
}

setup_dev() {
    print_header "Setting Up Development Environment"
    install_deps
    mkdir -p data/cache results analysis logs
    print_status "Development environment setup complete!"
    print_warning "To activate the virtual environment in new terminals, run: source venv/bin/activate"
}

quick_test() {
    print_header "Running Quick Tests"
    python -m pytest tests/ -v -x --tb=short
}

test_coverage() {
    print_header "Running Tests with Coverage"
    python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
}

check_quality() {
    print_header "Running Code Quality Checks"
    run_lint
    format_code
    run_tests
    print_status "Code quality checks completed!"
}

show_help() {
    print_header "Available Commands"
    echo "Usage: ./scripts.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install      - Install dependencies"
    echo "  venv         - Create virtual environment only"
    echo "  test         - Run test suite"
    echo "  clean        - Clean generated files"
    echo "  clean-venv   - Remove virtual environment"
    echo "  example      - Run the SMA crossover example"
    echo "  lint         - Run linting checks"
    echo "  format       - Format code with black"
    echo "  setup-dev    - Setup development environment"
    echo "  quick-test   - Run quick tests"
    echo "  coverage     - Run tests with coverage"
    echo "  quality      - Run all quality checks"
    echo "  help         - Show this help message"
}

main() {
    case "${1:-help}" in
        "install")
            install_deps
            ;;
        "venv")
            create_venv
            ;;
        "test")
            run_tests
            ;;
        "clean")
            clean_files
            ;;
        "clean-venv")
            clean_venv
            ;;
        "example")
            run_example
            ;;
        "lint")
            run_lint
            ;;
        "format")
            format_code
            ;;
        "setup-dev")
            setup_dev
            ;;
        "quick-test")
            quick_test
            ;;
        "coverage")
            test_coverage
            ;;
        "quality")
            check_quality
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

main "$@"
