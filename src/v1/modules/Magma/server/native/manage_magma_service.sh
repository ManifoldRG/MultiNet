#!/bin/bash

# This script helps manage the Magma API service
# You can use it to start, stop, restart, or check the status of the service

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Make sure the script has execute permissions
chmod +x run_magma_api.sh

# Function to display usage
usage() {
    echo -e "${YELLOW}Usage:${NC} $0 {setup-conda|install|start|stop|restart|status|logs}"
    echo "  setup-conda - Create and configure the conda environment"
    echo "  install  - Set up the service to run on system startup"
    echo "  start    - Start the Magma API service"
    echo "  stop     - Stop the Magma API service"
    echo "  restart  - Restart the Magma API service"
    echo "  status   - Check the status of the Magma API service"
    echo "  logs     - Show the service logs"
    exit 1
}

# Check if running with sudo
check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}Please run as root or with sudo${NC}"
        exit 1
    fi
}

# Function to setup conda environment
setup_conda_env() {
    echo -e "${YELLOW}Setting up conda environment for Magma API service...${NC}"
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}Conda is not installed or not in PATH.${NC}"
        echo "Please install Miniconda or Anaconda first."
        exit 1
    fi
    
    # Source conda for the current shell session
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo -e "${RED}Could not find conda.sh. Conda might not be properly installed.${NC}"
        exit 1
    fi
    
    # Check if the environment already exists
    if conda env list | grep -q "^magma "; then
        echo -e "${YELLOW}Conda environment 'magma' already exists.${NC}"
        echo -e "Updating packages..."
        conda activate magma
    else
        # Create the conda environment
        echo -e "Creating new conda environment 'magma'..."
        conda create -n magma python=3.10 -y
        conda activate magma
    fi
    
    # Clean installation approach - first PyTorch, then the rest
    cd ../..
    
    # First install PyTorch (needed for flash-attn)
    echo -e "${YELLOW}Installing PyTorch first...${NC}"
    pip install torch torchvision
    
    # Then install the Magma package with all dependencies
    echo -e "${YELLOW}Installing package with all dependencies...${NC}"
    pip install -e ".[server]" || { echo -e "${YELLOW}Some dependencies may not have installed properly${NC}"; }
    
    cd server/native
    
    echo -e "${GREEN}Conda environment 'magma' is ready.${NC}"
    echo -e "You can activate it manually with: ${YELLOW}conda activate magma${NC}"
}

# Function to install the service
install_service() {
    check_sudo
    
    echo -e "${YELLOW}Installing Magma API service...${NC}"
    
    # Get the absolute path to the service file
    SERVICE_PATH=$(readlink -f "$(dirname "$0")/magma-api.service")
    
    # Copy the service file to the systemd directory
    cp "$SERVICE_PATH" /etc/systemd/system/
    
    # Reload systemd to recognize the new service
    systemctl daemon-reload
    
    # Enable the service to start on boot
    systemctl enable magma-api.service
    
    echo -e "${GREEN}Service installed successfully.${NC}"
    echo -e "${YELLOW}You can now start the service with:${NC} sudo systemctl start magma-api.service"
}

# Main logic based on arguments
case "$1" in
    setup-conda)
        setup_conda_env
        ;;
    install)
        install_service
        ;;
    start)
        check_sudo
        systemctl start magma-api.service
        echo -e "${GREEN}Magma API service started.${NC}"
        ;;
    stop)
        check_sudo
        systemctl stop magma-api.service
        echo -e "${YELLOW}Magma API service stopped.${NC}"
        ;;
    restart)
        check_sudo
        systemctl restart magma-api.service
        echo -e "${GREEN}Magma API service restarted.${NC}"
        ;;
    status)
        systemctl status magma-api.service
        ;;
    logs)
        journalctl -u magma-api.service -f
        ;;
    *)
        usage
        ;;
esac

exit 0