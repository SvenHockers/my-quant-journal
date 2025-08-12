#!/bin/bash

# Usage: ./run_backtest.sh [config_file]
# The script automatically detects the strategy from the configuration file

set -e

# Default values
DEFAULT_CONFIG="config/config.yaml"
PYTHON_SCRIPT="examples/run_backtest.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    echo "Usage: $0 [config_file]"
    echo ""
    echo "Arguments:"
    echo "  config_file  Configuration file path (default: $DEFAULT_CONFIG)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with default config"
    echo "  $0 config/my_strategy.yaml           # Run with custom config"
    echo "  $0 config/regime_aware_config.yaml   # Run regime aware strategy"
    echo "  $0 config/sma_crossover_config.yaml  # Run SMA crossover strategy"
    echo ""
    echo "The script automatically detects the strategy from the config file."
    echo "Make sure your config file has a 'strategy.name' field set to:"
    echo "  - 'regime_aware' for Regime-Aware Portfolio Strategy"
    echo "  - 'sma_crossover' for SMA Crossover Strategy"
}

check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Virtual environment not detected. Activating..."
        if [[ -d "venv" ]]; then
            source venv/bin/activate
            print_success "Virtual environment activated"
        else
            print_error "Virtual environment not found. Please create one first:"
            echo "  python -m venv venv"
            echo "  source venv/bin/activate"
            exit 1
        fi
    fi
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    if ! python3 -c "import yaml" &> /dev/null; then
        print_error "PyYAML is not installed. Installing..."
        pip install PyYAML
    fi
    
    print_success "Dependencies check passed"
}

extract_strategy_from_config() {
    local config_file=$1
    
    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file not found: $config_file"
        exit 1
    fi
    
    # Extract strategy name using Python and yaml
    local strategy=$(python3 -c "
import yaml
import sys

try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    
    if 'strategy' in config and 'name' in config['strategy']:
        print(config['strategy']['name'])
    else:
        print('No strategy name found in config file')
        sys.exit(1)
        
except Exception as e:
    print(f'Error reading config file: {e}')
    sys.exit(1)
")
    
    if [[ $? -ne 0 ]]; then
        print_error "Failed to extract strategy from config file"
        exit 1
    fi
    
    echo "$strategy"
}

validate_strategy() {
    local config_file=$1
    print_info "Validating configuration via Python validator..."
    python3 scripts/validate_config.py validate --config "$config_file" | sed 's/^/[VALIDATION] /'
    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -ne 0 ]]; then
        print_error "Validation failed for: $config_file"
        exit $exit_code
    fi
}

run_backtest() {
    local config_file=$1
    local strategy=$2
    
    print_info "Running $strategy backtest with config: $config_file"
    
    cd "$(dirname "$0")"
    
    python3 "$PYTHON_SCRIPT" "$config_file"
    
    if [[ $? -eq 0 ]]; then
        print_success "Backtest completed successfully!"
    else
        print_error "Backtest failed!"
        exit 1
    fi
}

main() {
    print_info "Starting backtest runner..."
    
    CONFIG_FILE=${1:-$DEFAULT_CONFIG}
    
    # Check if config file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        echo ""
        echo "Please create a configuration file or use the example.yaml as a template:"
        echo "  cp example.yaml config/my_config.yaml"
        echo "  # Edit the config file, then run:"
        echo "  $0 config/my_config.yaml"
        exit 1
    fi
    
    check_venv
    check_dependencies
    
    # Extract and validate strategy from config file
    STRATEGY=$(extract_strategy_from_config "$CONFIG_FILE")
    print_info "Detected strategy: $STRATEGY"
    validate_strategy "$CONFIG_FILE"
    
    # Run the backtest
    run_backtest "$CONFIG_FILE" "$STRATEGY"
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

main "$@"
