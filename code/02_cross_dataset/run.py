"""Run or check this experiment; check mode is the default."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'common'))
from tahoe_repro.pipeline import experiment_main
if __name__=='__main__':
    raise SystemExit(experiment_main(HERE/'config.json'))
