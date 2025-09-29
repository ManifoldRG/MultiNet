#!/bin/bash

# This script provides a unified entry point for running Magma API service

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo -e "${YELLOW}Usage:${NC} $0 {docker|native|run} [command]"
    echo
    echo "Quick start:"
    echo "  run           - Run the server directly (installs dependencies if needed)"
    echo
    echo "Docker commands:"
    echo "  docker up     - Start the Docker container"
    echo "  docker down   - Stop the Docker container"
    echo "  docker logs   - View Docker container logs"
    echo "  docker build  - Build the Docker image"
    echo
    echo "Native service commands:"
    echo "  native setup    - Set up conda environment"
    echo "  native install  - Install as systemd service"
    echo "  native start    - Start the service"
    echo "  native stop     - Stop the service"
    echo "  native restart  - Restart the service"
    echo "  native status   - Check service status"
    echo "  native logs     - View service logs"
    echo "  native run      - Run directly without installing as a service"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$1" in
    run)
        # Simple run command to start the server directly
        cd "$(dirname "$0")"
        
        # Check if conda is available
        if ! command -v conda &> /dev/null; then
            echo -e "${RED}Conda is not installed or not in PATH${NC}"
            echo -e "${YELLOW}Please install Miniconda or Anaconda first:${NC}"
            echo -e "https://docs.conda.io/en/latest/miniconda.html"
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
        
        # Check if magma environment already exists
        if ! conda env list | grep -q "^magma "; then
            echo -e "${YELLOW}Creating conda environment 'magma'...${NC}"
            conda create -n magma python=3.10 -y || { echo -e "${RED}Failed to create conda environment${NC}"; exit 1; }
        fi
        
        # Activate the conda environment
        echo -e "${YELLOW}Activating conda environment 'magma'...${NC}"
        conda activate magma || { echo -e "${RED}Failed to activate conda environment${NC}"; exit 1; }
        
        # Clean installation approach - first PyTorch, then the rest
        echo -e "${GREEN}Installing Magma with dependencies...${NC}"
        cd ..
        
        # First install PyTorch (needed for flash-attn)
        echo -e "${YELLOW}Installing PyTorch first...${NC}"
        pip install torch torchvision
        
        # Then install the Magma package with all dependencies
        echo -e "${YELLOW}Installing package with all dependencies...${NC}"
        pip install -e ".[server]" || { echo -e "${YELLOW}Some dependencies may not have installed properly${NC}"; }
        
        cd server
        
        echo -e "${GREEN}Starting Magma API server...${NC}"
        python main.py
        ;;
    docker)
        cd "${SCRIPT_DIR}/docker"
        case "$2" in
            up)
                docker compose up -d
                ;;
            down)
                docker compose down
                ;;
            logs)
                docker compose logs -f
                ;;
            build)
                docker compose build
                ;;
            *)
                echo -e "${RED}Invalid docker command: ${2}${NC}"
                usage
                ;;
        esac
        ;;
    native)
        cd "${SCRIPT_DIR}/native"
        case "$2" in
            setup)
                ./manage_magma_service.sh setup-conda
                ;;
            install)
                sudo ./manage_magma_service.sh install
                ;;
            start)
                sudo ./manage_magma_service.sh start
                ;;
            stop)
                sudo ./manage_magma_service.sh stop
                ;;
            restart)
                sudo ./manage_magma_service.sh restart
                ;;
            status)
                ./manage_magma_service.sh status
                ;;
            logs)
                ./manage_magma_service.sh logs
                ;;
            run)
                chmod +x ./run_magma_api.sh
                ./run_magma_api.sh
                ;;
            *)
                echo -e "${RED}Invalid native command: ${2}${NC}"
                usage
                ;;
        esac
        ;;
    *)
        echo -e "${RED}Invalid deployment method: ${1}${NC}"
        usage
        ;;
esac
