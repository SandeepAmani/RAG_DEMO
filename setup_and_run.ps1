# =============================================================================
# SETUP AND RUN SCRIPT for RAG Demo (Windows PowerShell)
# =============================================================================
# Run this script in PowerShell to:
#   1. Install all Python dependencies
#   2. Prompt you for your Groq API key
#   3. Launch the RAG demo
#
# Usage:
#   Right-click this file → "Run with PowerShell"
#   OR open PowerShell in this folder and type:  .\setup_and_run.ps1
# =============================================================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  RAG DEMO — Setup and Launch" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Install dependencies
Write-Host "[1/3] Installing Python dependencies..." -ForegroundColor Yellow
pip install sentence-transformers numpy groq

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed. Make sure Python is installed." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[1/3] Dependencies installed successfully." -ForegroundColor Green

# Step 2: Prompt for Groq API key
Write-Host ""
Write-Host "[2/3] Groq API Key Setup" -ForegroundColor Yellow
Write-Host "  Get a free key at: https://console.groq.com" -ForegroundColor Gray

$apiKey = Read-Host "  Enter your GROQ_API_KEY"

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    Write-Host "ERROR: No API key entered." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Set the environment variable for this session
$env:GROQ_API_KEY = $apiKey
Write-Host "[2/3] API key set for this session." -ForegroundColor Green

# Step 3: Run the demo
Write-Host ""
Write-Host "[3/3] Launching RAG demo..." -ForegroundColor Yellow
Write-Host ""

python rag_demo.py

Write-Host ""
Read-Host "Press Enter to exit"
