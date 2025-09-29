#!/bin/bash

# WSL Proxy Setup Script
# Usage: source scripts/setup-wsl-proxy.sh --port <port_num>
# Example: source scripts/setup-wsl-proxy.sh --port 30080

set -e

# Default port
DEFAULT_PORT=30080
PORT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: source $0 --port <port_num>"
            echo "Sets up HTTP and HTTPS proxy environment variables for WSL"
            echo ""
            echo "Options:"
            echo "  --port <port_num>   Proxy port number (default: $DEFAULT_PORT)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Example:"
            echo "  source $0 --port 30080"
            return 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            return 1
            ;;
    esac
done

# Use default port if not specified
if [[ -z "$PORT" ]]; then
    PORT="$DEFAULT_PORT"
    echo "No port specified, using default port: $PORT"
fi

# Validate port number
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [[ "$PORT" -lt 1 ]] || [[ "$PORT" -gt 65535 ]]; then
    echo "Error: Invalid port number '$PORT'. Must be between 1 and 65535."
    return 1
fi

# Get Windows host IP address (default gateway)
HOST_IP=$(ip route show | grep -i default | awk '{ print $3}' | head -n1)

if [[ -z "$HOST_IP" ]]; then
    echo "Error: Could not determine Windows host IP address"
    return 1
fi

# Set proxy URL
PROXY_URL="http://${HOST_IP}:${PORT}"

# Test if proxy is accessible
echo "Testing proxy connectivity to $PROXY_URL..."
if curl -s --connect-timeout 3 --max-time 5 "$PROXY_URL" >/dev/null 2>&1; then
    echo "✓ Proxy server is accessible"
elif curl -s --connect-timeout 3 --max-time 5 -I "$PROXY_URL" >/dev/null 2>&1; then
    echo "✓ Proxy server is accessible (responds to HEAD requests)"
else
    echo "⚠ Warning: Cannot connect to proxy server at $PROXY_URL"
    echo "  The proxy may not be running or may not accept connections from WSL"
    echo "  Proceeding with environment variable setup anyway..."
fi

# Export proxy environment variables
export http_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"

# Set no_proxy for local addresses
export no_proxy="localhost,127.0.0.1,::1,${HOST_IP}"
export NO_PROXY="localhost,127.0.0.1,::1,${HOST_IP}"

echo ""
echo "✓ Proxy environment variables set:"
echo "  http_proxy=$http_proxy"
echo "  https_proxy=$https_proxy"
echo "  no_proxy=$no_proxy"
echo ""
echo "To unset proxy variables, run:"
echo "  unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY"