"""Optional figures from supplied result tables; no raw-data recomputation."""
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    module=runpy.run_path(str(ROOT/'code/common/scripts/reproduce_figures.py'))
    module['main'](['1', 'S1'])
