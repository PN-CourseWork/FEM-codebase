set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$script_dir"/ex_*.py; do
  echo "Running $(basename "$script")"
  uv run "$script"
done
