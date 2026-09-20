# Closed-loop ~1 hour smoke — thin wrapper around scripts/run_closed_loop_1h.py
#
# From evonav_env/:
#   powershell -ExecutionPolicy Bypass -File scripts/run_closed_loop_1h.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/run_closed_loop_1h.ps1 -Llm seed

param(
  [ValidateSet("groq", "seed", "ollama", "vllm")]
  [string]$Llm = "groq"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$pyArgs = @("scripts/run_closed_loop_1h.py", "--llm", $Llm)
if ($Llm -eq "seed") {
  $pyArgs += "--allow-seed-llm"
}

python @pyArgs
exit $LASTEXITCODE
