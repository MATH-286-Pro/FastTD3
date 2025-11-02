python - <<'PY'                                    
import torch, tensordict
print("torch =", torch.__version__)
print("tensordict =", tensordict.__version__)
from multiprocessing.reduction import ForkingPickler
print("ForkingPickler OK")
PY
