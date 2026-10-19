"""Digital proxies of reward, motivation and impulsivity (RM&I).

    snakemake -c4                                   # real data
    snakemake -c4 --configfile config/test.yaml     # synthetic smoke test
"""
import glob
import os
import re

configfile: "config/config.yaml"

SL = config["studentlife_dir"]
DEP = config["depresjon_dir"]
OUT = config["out_dir"]
SRC = os.path.join(workflow.basedir, "src")


def studentlife_uids():
    files = glob.glob(os.path.join(SL, "sensing", "gps", "gps_u*.csv"))
    return sorted(re.search(r"(u\d+)\.csv$", f).group(1) for f in files)


UIDS = studentlife_uids()
if not UIDS:
    raise WorkflowError(f"No GPS files found under {SL}/sensing/gps. Check studentlife_dir in config.")


rule all:
    input:
        f"{OUT}/report.md",


rule sl_features:
    """Person-day features for one participant; parallelises over 48 people."""
    input:
        gps=os.path.join(SL, "sensing", "gps", "gps_{uid}.csv"),
    output:
        daily=f"{OUT}/interim/features/{{uid}}.csv",
        person=f"{OUT}/interim/features/{{uid}}.person.json",
    params:
        src=SRC, sl_dir=SL, cfg=config,
    script:
        "workflow/scripts/sl_features.py"


rule sl_panel:
    """Merge features with daily EMA outcomes, apply coverage gate, add lead outcome."""
    input:
        daily=expand(f"{OUT}/interim/features/{{uid}}.csv", uid=UIDS),
        person=expand(f"{OUT}/interim/features/{{uid}}.person.json", uid=UIDS),
    output:
        panel=f"{OUT}/processed/panel.csv",
        person=f"{OUT}/processed/person.csv",
        summary=f"{OUT}/processed/data_summary.json",
    params:
        src=SRC, sl_dir=SL, cfg=config,
    script:
        "workflow/scripts/sl_panel.py"


rule mixed_models:
    input:
        panel=rules.sl_panel.output.panel,
    output:
        f"{OUT}/tables/mixed_models.csv",
    params:
        src=SRC, cfg=config,
    script:
        "workflow/scripts/mixed_models.py"


rule phq_anchor:
    input:
        person=rules.sl_panel.output.person,
    output:
        f"{OUT}/tables/phq_anchor.csv",
    params:
        src=SRC, cfg=config,
    script:
        "workflow/scripts/phq_anchor.py"


rule ml_loso:
    input:
        panel=rules.sl_panel.output.panel,
    output:
        f"{OUT}/tables/ml_cv_comparison.csv",
    params:
        src=SRC, cfg=config,
    threads: 4
    script:
        "workflow/scripts/ml_loso.py"


rule depresjon:
    input:
        scores=os.path.join(DEP, "scores.csv"),
    output:
        metrics=f"{OUT}/tables/depresjon_rhythm_metrics.csv",
        tests=f"{OUT}/tables/depresjon_tests.csv",
    params:
        src=SRC, dep_dir=DEP,
    script:
        "workflow/scripts/depresjon.py"


rule report:
    input:
        summary=rules.sl_panel.output.summary,
        mixed=rules.mixed_models.output[0],
        phq=rules.phq_anchor.output[0],
        ml=rules.ml_loso.output[0],
        dep_tests=rules.depresjon.output.tests,
        construct="config/construct_validity.tsv",
    output:
        f"{OUT}/report.md",
    script:
        "workflow/scripts/report.py"
