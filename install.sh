#!/usr/bin/env bash
#
# ML Bot — Production-oriented installer for Ubuntu 22.04/24.04
# Installs Python, dependencies, configures .env, and optionally systemd services.
#
set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PATH="$REPO_ROOT"
VENV_PATH="$REPO_ROOT/.venv"
ENV_FILE="$REPO_ROOT/.env"
LOG_LEVEL_DEFAULT="INFO"
LOG_JSON_DEFAULT="true"
LIVE_CONFIRM_PHRASE="ENABLE LIVE TRADING"
MIN_PYTHON_VERSION="3.12"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log_info() { echo "[INFO] $*"; }
log_warn() { echo "[WARN] $*" >&2; }
log_err()  { echo "[ERR]  $*" >&2; }
log_step() { echo ""; echo "==> $*"; }

# -----------------------------------------------------------------------------
# PHASE 1 — PRECHECKS
# -----------------------------------------------------------------------------
phase_prechecks() {
    log_step "Phase 1: Prechecks"

    # Must be Linux
    if [[ "$(uname -s)" != "Linux" ]]; then
        log_err "This installer is for Linux (Ubuntu 22.04/24.04). Detected: $(uname -s)"
        exit 1
    fi

    # Prefer Ubuntu detection
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        if [[ "${ID:-}" != "ubuntu" ]] && [[ "${ID_LIKE:-}" != *"ubuntu"* ]]; then
            log_warn "This installer targets Ubuntu. You are on: ${ID:-unknown}"
        fi
    fi

    # Verify we're in repo root
    if [[ ! -f "$REPO_ROOT/pyproject.toml" ]] || [[ ! -d "$REPO_ROOT/src/trading" ]]; then
        log_err "Must run from ml-bot repository root (contains pyproject.toml and src/trading)"
        exit 1
    fi

    # Verify systemctl if we'll install services
    if command -v systemctl &>/dev/null; then
        log_info "systemctl found"
    else
        log_warn "systemctl not found; systemd service installation will be skipped"
    fi

    # Check sudo availability (needed for apt and systemd)
    if [[ "$(id -u)" -eq 0 ]]; then
        log_warn "Running as root; consider using a non-root user with sudo"
        SUDO=""
    elif command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
        SUDO="sudo"
        log_info "sudo available"
    else
        log_warn "sudo may be required for apt and systemd; you may be prompted"
        SUDO="sudo"
    fi

    log_info "Prechecks passed"
}

# -----------------------------------------------------------------------------
# PHASE 2 — PYTHON + ENV SETUP
# -----------------------------------------------------------------------------
phase_python_setup() {
    log_step "Phase 2: Python and virtual environment"

    # Install apt packages if missing (python3.12 for Ubuntu 24.04; 22.04 may need deadsnakes PPA)
    local pkgs=(python3 python3-venv python3-pip build-essential curl git python3.12 python3.12-venv)
    local to_install=()
    for p in "${pkgs[@]}"; do
        if ! dpkg -s "$p" &>/dev/null; then
            to_install+=("$p")
        fi
    done
    if [[ ${#to_install[@]} -gt 0 ]]; then
        log_info "Installing packages: ${to_install[*]}"
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq "${to_install[@]}"
    else
        log_info "Required apt packages already installed"
    fi

    # Find Python 3.12+
    local py_cmd=""
    for v in 3.12 3.13 3.14; do
        if command -v "python${v}" &>/dev/null; then
            py_cmd="python${v}"
            break
        fi
    done
    if [[ -z "$py_cmd" ]] && command -v python3 &>/dev/null; then
        local ver
        ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
        if [[ -n "$ver" ]] && [[ "$(echo "$ver" | cut -d. -f1)" -ge 3 ]] && [[ "$(echo "$ver" | cut -d. -f2)" -ge 12 ]]; then
            py_cmd="python3"
        fi
    fi
    if [[ -z "$py_cmd" ]]; then
        log_err "Python ${MIN_PYTHON_VERSION}+ required."
        log_err "Ubuntu 24.04: sudo apt install python3.12 python3.12-venv"
        log_err "Ubuntu 22.04: add deadsnakes PPA, then sudo apt install python3.12 python3.12-venv"
        exit 1
    fi
    log_info "Using Python: $($py_cmd --version 2>&1)"

    # Create venv
    if [[ ! -d "$VENV_PATH" ]]; then
        log_info "Creating virtualenv at $VENV_PATH"
        "$py_cmd" -m venv "$VENV_PATH"
    else
        log_info "Virtualenv already exists at $VENV_PATH"
    fi

    # Upgrade pip/setuptools/wheel
    log_info "Upgrading pip, setuptools, wheel"
    "$VENV_PATH/bin/pip" install -q --upgrade pip setuptools wheel

    # Install project (with runtime deps; dev optional for production)
    log_info "Installing ml-bot package"
    "$VENV_PATH/bin/pip" install -q -e ".[dev]"

    log_info "Python setup complete"
}

# -----------------------------------------------------------------------------
# PHASE 3 — CONFIGURATION (calls bootstrap_env.py)
# -----------------------------------------------------------------------------
phase_configuration() {
    log_step "Phase 3: Configuration"

    if [[ -f "$ENV_FILE" ]]; then
        local backup="${ENV_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
        log_info "Backing up existing .env to $backup"
        cp -a "$ENV_FILE" "$backup"
    fi

    log_info "Running interactive configuration..."
    "$VENV_PATH/bin/python" "$REPO_ROOT/scripts/bootstrap_env.py" \
        --output "$ENV_FILE" \
        --install-path "$REPO_ROOT" \
        --default-mode paper \
        --default-dry-run true

    if [[ ! -f "$ENV_FILE" ]]; then
        log_err "Configuration did not produce .env file"
        exit 1
    fi
    chmod 600 "$ENV_FILE"
    log_info "Configuration written to $ENV_FILE (mode 600)"
}

# -----------------------------------------------------------------------------
# PHASE 4 — DIRECTORY SETUP
# -----------------------------------------------------------------------------
phase_directories() {
    log_step "Phase 4: Directory setup"

    for d in logs data "data/archive" "data/archive/session_summaries"; do
        local p="$REPO_ROOT/$d"
        if [[ ! -d "$p" ]]; then
            mkdir -p "$p"
            log_info "Created $p"
        fi
    done
    log_info "Directories ready"
}

# -----------------------------------------------------------------------------
# PHASE 5 — SYSTEMD SERVICES
# -----------------------------------------------------------------------------
phase_services() {
    log_step "Phase 5: Systemd services"

    if ! command -v systemctl &>/dev/null; then
        log_warn "Skipping systemd: systemctl not found"
        return
    fi

    local mode
    mode=$(grep -E "^TRADING_MODE=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- || echo "paper")
    mode="${mode:-paper}"

    local unit_name="trading-bot-${mode}.service"
    local unit_src="$REPO_ROOT/infra/systemd/${unit_name}"
    local unit_dst="/etc/systemd/system/${unit_name}"

    if [[ ! -f "$unit_src" ]]; then
        log_warn "Service file not found: $unit_src"
        log_info "Available: trading-bot-paper.service, trading-bot-demo.service, trading-bot-live.service"
        return
    fi

    local service_user
    service_user=$(whoami)
    read -r -p "Run service as user [$service_user]: " input_user
    service_user="${input_user:-$service_user}"

    log_info "Installing $unit_name (user=$service_user)"
    sed -e "s|__REPO_ROOT__|$REPO_ROOT|g" -e "s|__SERVICE_USER__|$service_user|g" "$unit_src" | $SUDO tee "$unit_dst" > /dev/null
    $SUDO systemctl daemon-reload

    echo ""
    echo "Service installed. Choose one:"
    echo "  1) Enable and start now"
    echo "  2) Enable only (start manually later)"
    echo "  3) Install only (do not enable)"
    read -r -p "Choice [1/2/3]: " choice
    choice="${choice:-3}"

    case "$choice" in
        1)
            $SUDO systemctl enable "$unit_name"
            $SUDO systemctl start "$unit_name"
            log_info "Service enabled and started"
            ;;
        2)
            $SUDO systemctl enable "$unit_name"
            log_info "Service enabled (not started)"
            ;;
        3)
            log_info "Service installed (not enabled)"
            ;;
        *)
            log_warn "Invalid choice; service installed only"
            ;;
    esac

    echo ""
    echo "Commands:"
    echo "  systemctl status $unit_name"
    echo "  systemctl start  $unit_name"
    echo "  systemctl stop   $unit_name"
    echo "  systemctl restart $unit_name"
    echo "  journalctl -u $unit_name -f"
}

# -----------------------------------------------------------------------------
# PHASE 6 — POST-INSTALL SUMMARY
# -----------------------------------------------------------------------------
phase_summary() {
    log_step "Phase 6: Post-install summary"

    local mode
    mode=$(grep -E "^TRADING_MODE=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- || echo "paper")
    local dry_run
    dry_run=$(grep -E "^TRADING_DRY_RUN=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- || echo "true")
    local placement="disabled"
    if [[ "$dry_run" == "false" ]]; then
        placement="ENABLED"
    fi

    echo ""
    echo "=============================================="
    echo "  ML Bot installation complete"
    echo "=============================================="
    echo ""
    echo "  Mode:              $mode"
    echo "  Order placement:   $placement"
    echo "  Env file:         $ENV_FILE"
    echo "  Repo path:        $REPO_ROOT"
    echo ""
    if command -v systemctl &>/dev/null; then
        local mode
        mode=$(grep -E "^TRADING_MODE=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- || echo "paper")
        echo "  Service:          trading-bot-${mode}.service"
        echo ""
    fi
    echo "  Key commands:"
    echo "    source $VENV_PATH/bin/activate"
    echo "    python -m trading.main"
    echo "    trading-bot"
    echo ""
    if command -v systemctl &>/dev/null; then
        local mode
        mode=$(grep -E "^TRADING_MODE=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- || echo "paper")
        echo "    systemctl status trading-bot-${mode}.service"
        echo "    journalctl -u trading-bot-${mode}.service -f"
        echo ""
    fi
    echo "=============================================="
}

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
main() {
    echo ""
    echo "ML Bot — Production Installer"
    echo "Target: Ubuntu 22.04/24.04"
    echo ""

    echo "Install path: $REPO_ROOT"
    read -r -p "Use this path? [Y/n]: " confirm
    if [[ "${confirm,,}" == "n" ]] || [[ "${confirm,,}" == "no" ]]; then
        log_err "Installation cancelled. Run from desired directory."
        exit 1
    fi

    phase_prechecks
    phase_python_setup
    phase_directories
    phase_configuration
    phase_services
    phase_summary
}

main "$@"
