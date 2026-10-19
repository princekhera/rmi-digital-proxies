import sys; sys.path.insert(0, snakemake.params.src)  # noqa: E702
import pandas as pd
from rmi.models import run_mixed_models

cfg = snakemake.params.cfg
panel = pd.read_csv(snakemake.input.panel)
res = run_mixed_models(panel, cfg["features"], cfg["outcomes"], random_slope=cfg["random_slope"])
res.to_csv(snakemake.output[0], index=False)
