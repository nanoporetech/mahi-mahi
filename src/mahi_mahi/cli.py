#!/usr/bin/env python

import argparse
import logging
import os
import re
import shlex
import sys
from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from .config import MahiMahiConfig, default_config_path, load_config
from .errors import UserFacingError
from .utils import get_mahi_mahi_version

# Locate the built-in config once at import time
DEFAULT_CFG_PATH = default_config_path()
BUILTIN_LOCI_FILES = {
    "hg19": resources.files("mahi_mahi.data").joinpath("str_loci.hg19.bed"),
    "hg38": resources.files("mahi_mahi.data").joinpath("str_loci.hg38.bed"),
}

# A sample name is used to build output filenames, so it must be a plain filename stem.
SAMPLE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9@._-]*")
SAMPLE_NAME_ERROR = "Sample name must be a simple filename stem (allowed characters: A-Za-z0-9@._-).  For full paths, use --out-dir."
ELEVATED_PRIVILEGES_WARNING = (
    "Mahi-mahi is running with elevated privileges. Run as a normal user without sudo or administrator privileges "
    "to reduce the risk of unintended changes to protected files."
)
PRIVILEGE_CHECK_FAILURE_WARNING = "Unable to determine whether Mahi-mahi is running with elevated privileges; continuing execution."


def warn_if_elevated() -> None:
    """Warn if Mahi-mahi is running as root, without preventing execution."""
    try:
        is_elevated = os.geteuid() == 0
    except Exception:
        logging.warning(PRIVILEGE_CHECK_FAILURE_WARNING)
        return
    if is_elevated:
        logging.warning(ELEVATED_PRIVILEGES_WARNING)


def validate_sample_name(sample: str) -> str:
    """Return the sample name if it is usable as a filename stem, otherwise raise."""
    if ".." in sample or not SAMPLE_NAME_PATTERN.fullmatch(sample):
        raise argparse.ArgumentTypeError(SAMPLE_NAME_ERROR)
    return sample


def prepare_output_dir(out_dir: Path, overwrite: bool) -> Path:
    """Create (if needed) and validate the output directory, returning its resolved path.

    Unless overwrite is set, the directory must be empty or not yet exist.  As well as protecting
    existing results, this means none of the paths mahi-mahi later writes can be a pre-existing
    symlink pointing outside the output directory.
    """
    if out_dir.exists() and not out_dir.is_dir():
        raise UserFacingError(f"Output path is not a directory: {out_dir}")
    if not overwrite and out_dir.is_dir() and any(out_dir.iterdir()):
        raise UserFacingError(
            f"Output directory is not empty: {out_dir}. Choose a new --out-dir, delete the existing output, or rerun with --overwrite."
        )
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UserFacingError(f"Cannot create or write to output directory {out_dir}: {exc.strerror or exc}") from exc
    resolved = out_dir.resolve()
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise UserFacingError(f"Output directory is not writable: {resolved}")
    return resolved


def safe_output_path(out_dir: Path, sample: str, suffix: str) -> Path:
    """Build an output path from a validated sample stem, guaranteed to stay inside out_dir."""
    if ".." in sample or not SAMPLE_NAME_PATTERN.fullmatch(sample):
        raise UserFacingError(SAMPLE_NAME_ERROR)
    resolved_dir = out_dir.resolve()
    candidate = (resolved_dir / f"{sample}{suffix}").resolve()
    if not candidate.is_relative_to(resolved_dir):
        raise UserFacingError(f"Refusing to write output outside the output directory: {candidate}")
    return candidate


def claim_log_file(out_dir: Path, sample: str) -> None:
    """Create the log file exclusively, so two concurrent runs cannot both write this sample's outputs.

    O_EXCL also fails on a symlink, so this rejects one planted since prepare_output_dir looked.
    """
    log_path = safe_output_path(out_dir, sample, ".log")
    try:
        os.close(os.open(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
    except FileExistsError as exc:
        raise UserFacingError(f"Output already exists: {log_path}. Choose a new --out-dir or --sample, or rerun with --overwrite.") from exc
    except OSError as exc:
        raise UserFacingError(f"Cannot write output file {log_path}: {exc.strerror or exc}") from exc


def resolve_loci_path(loci_arg: str) -> Path:
    """Resolve a loci argument to a filesystem Path, supporting built-in shortcuts."""
    builtin_path = BUILTIN_LOCI_FILES.get(loci_arg.lower())
    if builtin_path is not None:
        logging.info("Using built-in loci file for genome %s: %s", loci_arg, builtin_path)
        return Path(str(builtin_path))

    path = Path(loci_arg)
    if not path.exists():
        raise UserFacingError(f"Loci file not found: {loci_arg}")
    logging.info("Using loci file: %s", path)
    return path


def validate_input_paths(args: argparse.Namespace) -> None:
    """Validate alignment and reference files before analysis starts."""
    alignment_file = args.cram or args.bam
    alignment_option = "--cram" if args.cram else "--bam"
    alignment_path = Path(alignment_file)
    if not alignment_path.is_file():
        raise UserFacingError(f"Alignment file not found for {alignment_option}: {alignment_file}")

    index_suffix = ".crai" if args.cram else ".bai"
    index_path = Path(f"{alignment_path}{index_suffix}")
    if not index_path.is_file():
        raise UserFacingError(f"Alignment index not found for {alignment_option}: expected {index_path}")

    reference_path = Path(args.reference_genome_fasta)
    if not reference_path.is_file():
        raise UserFacingError(f"Reference file not found for --ref: {reference_path}")
    reference_index_path = Path(f"{reference_path}.fai")
    if not reference_index_path.is_file():
        raise UserFacingError(f"Reference index not found for --ref: expected {reference_index_path}")


def setup_file_logging(out_dir: Path, sample: str, overwrite: bool) -> None:
    """
    Add a FileHandler to duplicate logs to a log file inside the output directory.
    """
    # Appending, so that the empty file claim_log_file exclusively created is reused rather than reopened.
    file_handler = logging.FileHandler(str(safe_output_path(out_dir, sample, ".log")), mode="w" if overwrite else "a")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)


def log_traceback_to_file() -> None:
    root_logger = logging.getLogger()
    record = root_logger.makeRecord(root_logger.name, logging.ERROR, __file__, 0, "Error traceback", (), sys.exc_info())
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)


def write_resolved_config(out_dir: Path, sample: str, cfg: MahiMahiConfig) -> Path:
    resolved_config_path = safe_output_path(out_dir, sample, ".config.yaml")
    resolved_config = {
        "defaults": cfg.model_dump(),
        "profiles": {},
    }
    resolved_config_path.write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    return resolved_config_path


def _override_validation_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    path = ".".join(str(part) for part in error["loc"])
    if error["type"] == "extra_forbidden":
        return f"Config option {path} does not exist."
    return f"Invalid config override {path}: {error['msg']}"


def _config_validation_error(path: str | Path, exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"])
    return f"Invalid config in {path}: {location}: {error['msg']}"


def process_args_config() -> tuple[argparse.Namespace, MahiMahiConfig]:
    """Process command-line arguments and configuration file, and start logger.

    Returns:
        A tuple of (args, config) where args is the parsed command-line arguments
        and config is the loaded typed configuration."""
    parser = argparse.ArgumentParser(
        description=f"mahi-mahi v{get_mahi_mahi_version()}: Short Tandem Repeat analysis from nanopore data."
        " Arguments *not* in [] are required. See README.md for more detailed help.",
        epilog="For research use only.",
    )
    parser.add_argument("--version", action="version", version=get_mahi_mahi_version())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bam", help="BAM file")
    group.add_argument("--cram", help="CRAM file")

    parser.add_argument("--ref", required=True, dest="reference_genome_fasta", help="Reference genome file (FASTA format)")
    parser.add_argument(
        "--loci",
        required=True,
        dest="bed",
        help="BED file with STR loci, or built-in STR loci shortcut (hg19 or hg38)",
        type=str,
    )
    parser.add_argument(
        "--sample",
        required=True,
        dest="sample",
        type=validate_sample_name,
        help="Sample name to use in output VCF file, and as the stem of every output filename",
    )
    parser.add_argument(
        "--sex",
        required=True,
        dest="sex",
        choices={"XY", "XX"},
        help="Sex of sample",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        required=True,
        dest="out_dir",
        type=Path,
        help="Path to output directory",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        default=DEFAULT_CFG_PATH,
        help="Path to mahi-mahi YAML config file. "
        "Defaults to the built-in file shipped with mahi-mahi "
        f"({Path(str(DEFAULT_CFG_PATH)).relative_to(Path(__file__).parents[2])}).",
    )
    parser.add_argument(
        "--profile",
        required=False,
        default=None,
        help="Named config profile to apply after defaults.",
    )
    parser.add_argument(
        "--set",
        required=False,
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override a config value; repeat for multiple overrides. Examples: "
            "--set min_read_support_per_allele=6, --set strand_bias_method=counts_threshold, "
            "--set strand_bias_exclude_suspicious_reads=none."
        ),
    )
    parser.add_argument(
        "--repeat-id",
        required=False,
        dest="repeat_id",
        help="Repeat ID to analyze (omit to analyze all repeats in --loci)",
        type=str,
    )
    parser.add_argument(
        "--no-clustering",
        required=False,
        dest="no_clustering",
        action="store_true",
        help="Turns off clustering of reads into two alleles based on read alignment length, and instead uses existing haplotypes only",
    )
    parser.add_argument(
        "--debug",
        required=False,
        action="store_true",
        help="Enable debug messages for detailed logging",
    )
    parser.add_argument(
        "--bypass-haplotag-check",
        required=False,
        action="store_true",
        help="Accept non-haplotagged input.  Caution: will result in merging close-in-length alleles, and may more generally reduce accuracy",
    )
    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Allow --out-dir to be non-empty, overwriting any existing mahi-mahi output in it",
    )
    args = parser.parse_args()

    # Set logging level based on --debug flag
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level)
    warn_if_elevated()

    args.out_dir = prepare_output_dir(args.out_dir, args.overwrite)
    if not args.overwrite:
        claim_log_file(args.out_dir, args.sample)
    try:
        setup_file_logging(args.out_dir, args.sample, args.overwrite)
    except OSError as exc:
        raise UserFacingError(f"Cannot create or write to output directory {args.out_dir}: {exc.strerror or exc}") from exc
    logging.info("mahi-mahi version: %s **FOR RESEARCH USE ONLY**", get_mahi_mahi_version())
    logging.info("Command used: %s", shlex.join(sys.argv))

    logging.info("Using config file: %s", args.config)
    if args.profile is not None:
        logging.info("Using config profile: %s", args.profile)
    try:
        cfg = load_config(args.config, profile=args.profile)
    except ValidationError as exc:
        raise UserFacingError(_config_validation_error(args.config, exc)) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise UserFacingError(str(exc)) from exc
    if args.set:
        try:
            cfg = load_config(args.config, profile=args.profile, set_overrides=args.set)
        except ValidationError as exc:
            raise UserFacingError(_override_validation_error(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise UserFacingError(str(exc)) from exc
    try:
        resolved_config_path = write_resolved_config(args.out_dir, args.sample, cfg)
    except OSError as exc:
        raise UserFacingError(f"Cannot create or write to output directory {args.out_dir}: {exc.strerror or exc}") from exc
    logging.info("Saved resolved config to: %s", resolved_config_path)
    logging.info("Using strand bias method: %s", cfg.strand_bias_method)
    return args, cfg
