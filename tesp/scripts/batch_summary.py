#!/usr/bin/env python

"""Summarise all the jobs submitted within a given batch."""

from pathlib import Path

import click
import structlog
from structlog.processors import JSONRenderer

from tesp.luigi_db_utils import retrieve_status

COMMON_PROCESSORS = [
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="ISO"),
    JSONRenderer(sort_keys=True),
]


@click.command()
@click.option(
    "--indir",
    type=click.Path(file_okay=False, readable=True),
    help="The input directory of the batchjob.",
)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False, writable=True),
    help=(
        "The output directory to contain the done, failed, pending "
        "and running lists."
    ),
)
def main(indir, outdir):
    """ """
    # status lists
    reprocess = []
    done = []
    fail = []
    running = []

    indir = Path(indir)
    outdir = Path(outdir)
    files = indir.rglob("luigi-task-hist.db")

    msg = "package task state at end of batchjob"
    structlog.configure(processors=COMMON_PROCESSORS)
    state_log = structlog.get_logger("task-final-state")

    final_state_out_fname = outdir.joinpath(f"{indir.name}-level1-final-state.jsonl")
    with open(final_state_out_fname, "w") as fobj:
        structlog.configure(logger_factory=structlog.PrintLoggerFactory(fobj))

        for fname in files:
            done_df, fail_df, pending_df, running_df = retrieve_status(
                str(fname), "Package"
            )
            reprocess.extend(pending_df.value_level1.tolist())
            done.extend(done_df.value_level1.tolist())
            fail.extend(fail_df.value_level1.tolist())
            running.extend(running_df.value_level1.tolist())

            # log granules that have state DONE
            for i, row in done_df.iterrows():
                state_log.info(
                    msg,
                    final_state="done",
                    level1=row.value_level1,
                    granule=row.value_granule,
                )

            # log granules that have state PENDING
            for i, row in pending_df.iterrows():
                state_log.info(
                    msg,
                    final_state="pending",
                    level1=row.value_level1,
                    granule=row.value_granule,
                )

            # log granules that have state FAILED
            for i, row in fail_df.iterrows():
                state_log.info(
                    msg,
                    final_state="failed",
                    level1=row.value_level1,
                    granule=row.value_granule,
                )

            # log granules that have state RUNNING
            for i, row in running_df.iterrows():
                state_log.info(
                    msg,
                    final_state="running",
                    level1=row.value_level1,
                    granule=row.value_granule,
                )

    out_fname_fmt = "level-1-final_state-{}.txt"
    out_fname = outdir.joinpath(out_fname_fmt.format("pending"))
    with open(out_fname, "w") as src:
        src.writelines([f"{fname}\n" for fname in reprocess])

    out_fname = outdir.joinpath(out_fname_fmt.format("done"))
    with open(out_fname, "w") as src:
        src.writelines([f"{fname}\n" for fname in done])

    out_fname = outdir.joinpath(out_fname_fmt.format("failed"))
    with open(out_fname, "w") as src:
        src.writelines([f"{fname}\n" for fname in fail])

    out_fname = outdir.joinpath(out_fname_fmt.format("running"))
    with open(out_fname, "w") as src:
        src.writelines([f"{fname}\n" for fname in running])


if __name__ == "__main__":
    main()
