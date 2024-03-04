"""
Show expected processing/ancillary information for a Level1 path on the current system.

It will look for wagl/luigi config in the default locations, so if you have an existing
wagl environment set up (such as our NCI modules) you may only need to specify a level1 path.

It will output in a readable yaml format by default, but you can also specify `--paths-only`
to get a simpler list of pure paths (suitable for typical bash commands).
"""

from typing import List, Optional, Tuple

import click
from click import echo, style
from osgeo import osr

from wagl.acquisition import PackageIdentificationHint, acquisitions
from wagl.ancillary import AncillaryConfig, BrdfMode, find_needed_acquisition_ancillary
from wagl.brdf import AncillaryTier

osr.UseExceptions()


def find_needed_level1_ancillary(
    level1_paths: List[str],
    *,
    mode: BrdfMode = None,
    acq_parser_hint: PackageIdentificationHint = None,
    luigi_config_path: Optional[str] = None,
) -> Tuple[str, AncillaryTier, List[str]]:
    """
    Find the ancillary files needed for a level1 dataset.

    :param level1_paths: Path to level1 datasets
    :param acq_parser_hint: Hint on how to interpret the package ("s2_sinergise")
    :param luigi_config_path: Optional Path to luigi config file, if not loading from default places.
    :return: Tuple of (dataset-label, processing-tier, paths-needed)
    """
    for level1_path in level1_paths:
        container = acquisitions(str(level1_path), acq_parser_hint)
        acquisition = container.get_highest_resolution()[0][0]

        tiers, paths = find_needed_acquisition_ancillary(
            acquisition, AncillaryConfig.from_luigi(luigi_config_path), mode=mode
        )
        if len(tiers) != 1:
            raise ValueError(
                f"Expected one tier, got: {tiers!r} in {level1_path}. TODO: Should this happen?"
            )
        [tier] = tiers
        yield container.label, tier, paths


@click.command("ard_show", help=__doc__)
@click.argument("level1_paths", nargs=-1, type=str, required=True)
@click.option(
    "--mode",
    type=str,
    help="Brdf mode",
)
@click.option(
    "--acq-parser-hint",
    type=str,
    help="Optional acquisition parser hint (e.g. 's2_sinergise')",
)
@click.option(
    "--luigi-config-path",
    type=str,
    help="Optional path to a non-standard luigi config location",
)
@click.option(
    "-p",
    "--paths-only",
    is_flag=True,
    help="Output bare paths only, no yaml formatting",
)
def main(level1_paths, acq_parser_hint, mode, luigi_config_path, paths_only):
    for i, (label, tier, needed_paths) in enumerate(
        find_needed_level1_ancillary(
            level1_paths,
            acq_parser_hint=acq_parser_hint,
            luigi_config_path=luigi_config_path,
            mode=mode,
        )
    ):
        if paths_only:
            for path in needed_paths:
                echo(path)
        else:
            # Readable yaml output with everything.
            label = style(f"{label!r}", fg="blue")
            echo(f"dataset: {label}")
            echo(f"tier: {style(tier, bold=True)}")
            echo()
            for path in needed_paths:
                echo(f"- {path!r}")
            if i != 0:
                echo()


if __name__ == "__main__":
    main()
