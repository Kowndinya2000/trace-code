# Shared, relocatable defaults. Source this from a demo launcher.
FORT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRACE_ROOT=${TRACE_ROOT:-$(cd "$FORT/../.." && pwd)}
export TRACE_DATA=${TRACE_DATA:-$TRACE_ROOT/data}
export TRACE_RUNS=${TRACE_RUNS:-$TRACE_ROOT/runs}
PY=$(command -v "${PYTHON:-${TRACE_PYTHON:-python}}") || return 1
export TRACE_PYTHON="$PY"
_ENVBIN=$(dirname "$PY")
export PATH="$_ENVBIN:$PATH"
export LD_LIBRARY_PATH="$_ENVBIN/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CALIB=${TRACE_CALIB:-${CALIB:-$TRACE_DATA/calibration/camera_to_base.txt}}
MASKRCNN=${TRACE_MASKRCNN:-${MASKRCNN:-$TRACE_DATA/segmentation/maskrcnn.pth}}
export TRACE_CALIB="$CALIB" TRACE_MASKRCNN="$MASKRCNN"
mkdir -p "$TRACE_RUNS"
for required in "$CALIB" "$MASKRCNN"; do
  if [ ! -f "$required" ]; then
    echo "[demo] required local hardware input missing: $required (see docs/HARDWARE.md)" >&2
    return 1
  fi
done
