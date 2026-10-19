import sys; sys.path.insert(0, snakemake.params.src)  # noqa: E702
import json
from rmi.io_studentlife import read_csv_clean, sensor_path
from rmi.features import participant_features

uid, sl, cfg = snakemake.wildcards.uid, snakemake.params.sl_dir, snakemake.params.cfg
streams = {s: read_csv_clean(sensor_path(sl, s, uid))
           for s in ("gps", "activity", "conversation", "phonelock", "dark")}
daily, person = participant_features(streams, cfg)
daily.insert(0, "uid", uid)
daily.to_csv(snakemake.output.daily, index=False)
with open(snakemake.output.person, "w") as fh:
    json.dump({"uid": uid, **person}, fh)
