#!/usr/bin/env python

"""A temporary workflow for processing S2 data into an ARD package."""

import json
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import basename
from os.path import join as pjoin
from pathlib import Path
from typing import Optional, Sequence

import luigi
import yaml
from eodatasets3.wagl import Granule, package
from luigi.local_target import LocalFileSystem

from eugl import s2cl
from eugl.fmask import fmask
from eugl.gqa import GQATask
from tesp.constants import ProductPackage
from tesp.metadata import _get_tesp_metadata
from tesp.package import package_non_standard, write_stac_metadata
from wagl.acquisition import (
    PackageIdentificationHint,
    acquisitions,
    preliminary_acquisitions_data,
)
from wagl.logs import STATUS_LOGGER, TASK_LOGGER
from wagl.singlefile_workflow import DataStandardisation

QA_PRODUCTS = ["gqa", "fmask", "s2cloudless"]


@luigi.Task.event_handler(luigi.Event.FAILURE)
def on_failure(task, exception):
    """Capture any Task Failure here."""
    TASK_LOGGER.exception(
        event="task-failure",
        task=task.get_task_family(),
        params=task.to_str_params(),
        level1=getattr(task, "level1", ""),
        granule=getattr(task, "granule", ""),
        stack_info=True,
        status="failure",
        exception=exception.__str__(),
        traceback=traceback.format_exc().splitlines(),
    )


@luigi.Task.event_handler(luigi.Event.SUCCESS)
def on_success(task):
    """Capture any Task Success here."""
    TASK_LOGGER.info(
        event="task-success",
        task=task.get_task_family(),
        params=task.to_str_params(),
        level1=getattr(task, "level1", ""),
        granule=getattr(task, "granule", ""),
        status="success",
    )


class WorkDir(luigi.Task):
    """Initialises the working directory in a controlled manner.
    Alternatively this could be initialised upfront during the
    ARD Task submission phase.
    """

    level1: str = luigi.Parameter()
    workdir: str = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.workdir)

    def run(self):
        local_fs = LocalFileSystem()
        local_fs.mkdir(self.output().path)


class RunS2Cloudless(luigi.Task):
    """Execute the s2cloudless algorithm for a given granule."""

    level1: str = luigi.Parameter()
    granule: str = luigi.Parameter()
    workdir: str = luigi.Parameter()
    acq_parser_hint: PackageIdentificationHint = luigi.OptionalParameter(default="")
    threshold: float = luigi.FloatParameter(default=s2cl.THRESHOLD)
    average_over: int = luigi.IntParameter(default=s2cl.AVERAGE_OVER)
    dilation_size: int = luigi.IntParameter(default=s2cl.DILATION_SIZE)

    def platform_id(self):
        container = acquisitions(self.level1, hint=self.acq_parser_hint)
        sample_acq = container.get_all_acquisitions()[0]
        return sample_acq.platform_id

    def complete(self):
        if not self.platform_id().startswith("SENTINEL"):
            return True

        return super().complete()

    def output(self):
        if not self.platform_id().startswith("SENTINEL"):
            return None

        prob_out_fname = pjoin(self.workdir, f"{self.granule}.prob.s2cloudless.tif")
        mask_out_fname = pjoin(self.workdir, f"{self.granule}.mask.s2cloudless.tif")
        metadata_out_fname = pjoin(self.workdir, f"{self.granule}.s2cloudless.yaml")

        out_fnames = {
            "cloud_prob": luigi.LocalTarget(prob_out_fname),
            "cloud_mask": luigi.LocalTarget(mask_out_fname),
            "metadata": luigi.LocalTarget(metadata_out_fname),
        }

        return out_fnames

    def run(self):
        if not self.platform_id().startswith("SENTINEL"):
            return

        out_fnames = self.output()
        with out_fnames["cloud_prob"].temporary_path() as prob_out_fname:
            with out_fnames["cloud_mask"].temporary_path() as mask_out_fname:
                with out_fnames["metadata"].temporary_path() as metadata_out_fname:
                    s2cl.s2cloudless_processing(
                        self.level1,
                        self.granule,
                        prob_out_fname,
                        mask_out_fname,
                        metadata_out_fname,
                        self.workdir,
                        acq_parser_hint=self.acq_parser_hint,
                        threshold=self.threshold,
                        average_over=self.average_over,
                        dilation_size=self.dilation_size,
                    )


class S2Cloudless(luigi.Task):
    """Execute the Fmask algorithm for a given granule."""

    level1 = luigi.Parameter()
    workdir = luigi.Parameter()
    acq_parser_hint = luigi.OptionalParameter(default="")
    threshold = luigi.FloatParameter(default=s2cl.THRESHOLD)
    average_over = luigi.IntParameter(default=s2cl.AVERAGE_OVER)
    dilation_size = luigi.IntParameter(default=s2cl.DILATION_SIZE)

    def requires(self):
        # issues task per granule
        for granule in preliminary_acquisitions_data(self.level1, self.acq_parser_hint):
            yield RunS2Cloudless(
                self.level1,
                granule["id"],
                self.workdir,
                acq_parser_hint=self.acq_parser_hint,
                threshold=self.threshold,
                average_over=self.average_over,
                dilation_size=self.dilation_size,
            )


class RunFmask(luigi.Task):
    """Execute the Fmask algorithm for a given granule."""

    level1: str = luigi.Parameter()
    granule: str = luigi.Parameter()
    workdir: str = luigi.Parameter()
    cloud_buffer_distance: float = luigi.FloatParameter(default=150.0)
    cloud_shadow_buffer_distance: float = luigi.FloatParameter(default=300.0)
    parallax_test: bool = luigi.BoolParameter()
    upstream_settings: dict = luigi.DictParameter(default={})
    acq_parser_hint: PackageIdentificationHint = luigi.OptionalParameter(default="")

    def output(self):
        out_fname1 = pjoin(self.workdir, f"{self.granule}.fmask.img")
        out_fname2 = pjoin(self.workdir, f"{self.granule}.fmask.yaml")

        out_fnames = {
            "image": luigi.LocalTarget(out_fname1),
            "metadata": luigi.LocalTarget(out_fname2),
        }

        return out_fnames

    def run(self):
        out_fnames = self.output()
        with out_fnames["image"].temporary_path() as out_fname1:
            with out_fnames["metadata"].temporary_path() as out_fname2:
                fmask(
                    self.level1,
                    self.granule,
                    out_fname1,
                    out_fname2,
                    self.workdir,
                    self.acq_parser_hint,
                    self.cloud_buffer_distance,
                    self.cloud_shadow_buffer_distance,
                    self.parallax_test,
                )


# useful for testing fmask via the CLI
class Fmask(luigi.WrapperTask):
    """A helper task that issues RunFmask Tasks."""

    level1: str = luigi.Parameter()
    workdir: str = luigi.Parameter()
    cloud_buffer_distance: float = luigi.FloatParameter(default=150.0)
    cloud_shadow_buffer_distance: float = luigi.FloatParameter(default=300.0)
    parallax_test: bool = luigi.BoolParameter()
    acq_parser_hint: PackageIdentificationHint = luigi.OptionalParameter(default="")

    def requires(self):
        # issues task per granule
        for granule in preliminary_acquisitions_data(self.level1, self.acq_parser_hint):
            yield RunFmask(
                self.level1,
                granule["id"],
                self.workdir,
                self.cloud_buffer_distance,
                self.cloud_shadow_buffer_distance,
                self.parallax_test,
            )


class Package(luigi.Task):
    """Creates the final packaged product once wagl, Fmask
    and gqa have executed successfully.
    """

    level1: str = luigi.Parameter()
    workdir: str = luigi.Parameter()
    granule: str = luigi.OptionalParameter(default="")
    pkgdir: str = luigi.Parameter()
    yamls_dir: str = luigi.OptionalParameter(default="")
    cleanup: bool = luigi.BoolParameter()
    acq_parser_hint: PackageIdentificationHint = luigi.OptionalParameter(default="")
    products: Sequence[str] = luigi.ListParameter(default=ProductPackage.default())
    qa_products: Sequence[str] = luigi.ListParameter(default=QA_PRODUCTS)

    # fmask settings
    cloud_buffer_distance: float = luigi.FloatParameter(default=150.0)
    cloud_shadow_buffer_distance: float = luigi.FloatParameter(default=300.0)
    parallax_test: bool = luigi.BoolParameter()

    # s2cloudless settings
    threshold: float = luigi.FloatParameter(default=s2cl.THRESHOLD)
    average_over: int = luigi.IntParameter(default=s2cl.AVERAGE_OVER)
    dilation_size: int = luigi.IntParameter(default=s2cl.DILATION_SIZE)

    non_standard_packaging: bool = luigi.BoolParameter()
    product_maturity: Optional[str] = luigi.OptionalParameter(default="stable")

    # STAC
    stac_base_url: Optional[str] = luigi.OptionalParameter(default="")
    explorer_base_url: Optional[str] = luigi.OptionalParameter(default="")

    def requires(self):
        # Ensure configuration values are valid
        # self._validate_cfg()

        tasks = {
            "wagl": DataStandardisation(
                self.level1,
                self.workdir,
                self.granule,
                acq_parser_hint=self.acq_parser_hint,
            ),
            "fmask": RunFmask(
                self.level1,
                self.granule,
                self.workdir,
                self.cloud_buffer_distance,
                self.cloud_shadow_buffer_distance,
                self.parallax_test,
                acq_parser_hint=self.acq_parser_hint,
            ),
            "s2cloudless": RunS2Cloudless(
                self.level1,
                self.granule,
                self.workdir,
                acq_parser_hint=self.acq_parser_hint,
                threshold=self.threshold,
                average_over=self.average_over,
                dilation_size=self.dilation_size,
            ),
            "gqa": GQATask(
                level1=self.level1,
                acq_parser_hint=self.acq_parser_hint,
                granule=self.granule,
                workdir=self.workdir,
            ),
        }

        # Need to improve pluggability across tesp/eugl/wagl
        # and adopt patterns that facilitate reuse
        for key in list(tasks.keys()):
            if key != "wagl" and key not in list(self.qa_products):
                del tasks[key]

        return tasks

    def output(self):
        # temp work around. rather than duplicate the packaging logic
        # create a text file to act as a completion target
        # this could be changed to be a database record
        parent_dir = Path(self.workdir).parent
        out_fname = parent_dir.joinpath(f"{self.granule}.completed")

        return luigi.LocalTarget(str(out_fname))

    def run(self):
        def search_for_external_level1_metadata() -> Optional[Path]:
            if self.yamls_dir is None or self.yamls_dir == "":
                return None

            level1 = Path(self.level1)

            # Level1 is in a three-level directory structure, and we mirror it in the yaml_dir
            # like this:
            #     '{yaml_dir}/2021/2021-02/25S150E-30S155E/{yaml}'
            result = (
                Path(self.yamls_dir)
                / level1.parent.parent.parent.name
                / level1.parent.parent.name
                / level1.parent.name
                / (level1.stem + ".odc-metadata.yaml")
            )

            # If a singular yaml doesn't exist, there could be separate granule yamls
            if not result.exists():
                result = result.with_name(
                    f"{level1.stem}.{self.granule}.odc-metadata.yaml"
                )

            if not result.exists():
                raise ValueError(
                    "Could not find matching metadata for L1 in the given yaml directory."
                    f"Tried with and without granule in path: {result.as_posix()!r} "
                    f"for dataset {self.level1!r}. "
                    f"(if you intended to use a sibling yaml file, don't specify a yaml directory)"
                )

            return result

        # TODO; the package_file func can accept additional fnames for yamls etc
        wagl_fname = Path(self.input()["wagl"].path)
        fmask_img_fname = Path(self.input()["fmask"]["image"].path)
        fmask_doc_fname = Path(self.input()["fmask"]["metadata"].path)
        gqa_doc_fname = Path(self.input()["gqa"].path)

        if self.input()["s2cloudless"] is not None:
            s2cloudless_prob_fname = Path(
                self.input()["s2cloudless"]["cloud_prob"].path
            )
            s2cloudless_mask_fname = Path(
                self.input()["s2cloudless"]["cloud_mask"].path
            )
            s2cloudless_metadata_fname = Path(
                self.input()["s2cloudless"]["metadata"].path
            )
        else:
            s2cloudless_prob_fname = None
            s2cloudless_mask_fname = None
            s2cloudless_metadata_fname = None

        tesp_doc_fname = Path(self.workdir) / f"{self.granule}.tesp.yaml"
        with tesp_doc_fname.open("w") as src:
            yaml.safe_dump(_get_tesp_metadata(), src)

        md = {}
        for eods_granule in Granule.for_path(
            wagl_fname,
            granule_names=[self.granule],
            fmask_image_path=fmask_img_fname,
            fmask_doc_path=fmask_doc_fname,
            s2cloudless_prob_path=s2cloudless_prob_fname,
            s2cloudless_mask_path=s2cloudless_mask_fname,
            s2cloudless_doc_path=s2cloudless_metadata_fname,
            gqa_doc_path=gqa_doc_fname,
            tesp_doc_path=tesp_doc_fname,
            level1_metadata_path=search_for_external_level1_metadata(),
        ):
            if self.non_standard_packaging:
                ds_id, md_path = package_non_standard(Path(self.pkgdir), eods_granule)
            else:
                ds_id, md_path = package(
                    Path(self.pkgdir),
                    eods_granule,
                    product_maturity=self.product_maturity,
                    included_products=self.products,
                )

                if self.stac_base_url != "" and self.explorer_base_url != "":
                    write_stac_metadata(
                        md_path, self.pkgdir, self.stac_base_url, self.explorer_base_url
                    )

            md[ds_id] = md_path
            STATUS_LOGGER.info(
                "packaged dataset",
                granule=self.granule,
                level1=self.level1,
                dataset_id=str(ds_id),
                dataset_path=str(md_path),
            )

        if self.cleanup:
            shutil.rmtree(self.workdir)

        with self.output().temporary_path() as out_fname:
            with open(out_fname, "w") as outf:
                data = {
                    "params": self.to_str_params(),
                    # JSON can't serialise the returned Path obj
                    "packaged_datasets": {str(k): str(v) for k, v in md.items()},
                }
                json.dump(data, outf)


def list_packages(workdir, acq_parser_hint, pkgdir, yamls_dir):
    def worker(level1):
        work_root = pjoin(workdir, f"{basename(level1)}.ARD")

        result = []
        for granule in preliminary_acquisitions_data(level1, acq_parser_hint):
            work_dir = pjoin(work_root, granule["id"])
            if yamls_dir is None or yamls_dir == "":
                result.append(
                    Package(
                        level1,
                        work_dir,
                        granule["id"],
                        pkgdir,
                        acq_parser_hint=acq_parser_hint,
                    )
                )
            else:
                result.append(
                    Package(
                        level1,
                        work_dir,
                        granule["id"],
                        pkgdir,
                        acq_parser_hint=acq_parser_hint,
                        yamls_dir=yamls_dir,
                    )
                )

        return result

    return worker


class ARDP(luigi.WrapperTask):
    """A helper Task that issues Package Tasks for each Level-1
    dataset listed in the `level1_list` parameter.
    """

    level1_list: str = luigi.Parameter(
        description="A path to a file that contains a list of level1 paths. (confusingly named)"
    )

    workdir: str = luigi.Parameter()
    pkgdir: str = luigi.Parameter()
    acq_parser_hint: PackageIdentificationHint = luigi.OptionalParameter(default="")
    yamls_dir: str = luigi.OptionalParameter(default="")

    def requires(self):
        with open(self.level1_list) as src:
            level1_list = [level1.strip() for level1 in src.readlines()]

        worker = list_packages(
            self.workdir, self.acq_parser_hint, self.pkgdir, self.yamls_dir
        )

        executor = ThreadPoolExecutor()
        futures = [executor.submit(worker, level1) for level1 in level1_list]

        for future in as_completed(futures):
            yield from future.result()


if __name__ == "__main__":
    luigi.run()
