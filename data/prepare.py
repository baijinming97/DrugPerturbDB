"""Download, verify and prepare datasets for the four analysis workflows."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/common'))
from tahoe_repro.pipeline import data_main

if __name__=='__main__':
    raise SystemExit(data_main(ROOT))
