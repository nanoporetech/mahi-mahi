#!/usr/bin/env python
import logging
import shlex
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pysam

from mahi_mahi.cli import log_traceback_to_file, process_args_config, resolve_loci_path, safe_output_path, validate_input_paths
from mahi_mahi.errors import UserFacingError
from mahi_mahi.io_utils.bam_utils import write_tagged_bam
from mahi_mahi.io_utils.bed_utils import load_loci_list, merge_nearby_loci
from mahi_mahi.io_utils.tsv_utils import write_tsv
from mahi_mahi.io_utils.vcf_utils import filter_loci_for_output, write_vcf
from mahi_mahi.locus_validation import validate_merged_loci
from mahi_mahi.methylation import bam_has_methylation
from mahi_mahi.region_processing import process_regions
from mahi_mahi.stats import is_haplotagged
from mahi_mahi.utils import check_bam_reference_compatibility, convert_cram_to_bam, detect_unsplit_2d_reads


def main() -> None:
    try:
        _run()
    except UserFacingError as exc:
        log_traceback_to_file()
        logging.error("%s", exc)
        if "--debug" in sys.argv:
            raise
        raise SystemExit(1) from None
    except Exception:
        log_traceback_to_file()
        if "--debug" in sys.argv:
            raise
        logging.error("Unexpected error occurred. See the output log for details.")
        raise SystemExit(1) from None


def _run() -> None:
    args, cfg = process_args_config()
    validate_input_paths(args)

    # Parse regions bed file
    merge_dist = cfg.region_merge_distance
    bed_file = resolve_loci_path(args.bed)
    unmerged_loci = load_loci_list(bed_file, args.repeat_id)
    merged_loci = merge_nearby_loci(unmerged_loci, merge_dist)
    validate_merged_loci(merged_loci)

    if len(merged_loci) == 0:
        raise UserFacingError("No valid loci found in the provided BED file.")
    logging.info("Loading alignments")
    input_alignment = args.cram or args.bam
    chroms_to_check = {locus.chrom for locus in merged_loci}
    check_bam_reference_compatibility(input_alignment, args.reference_genome_fasta, chroms_to_check)
    with tempfile.TemporaryDirectory() as temp_folder:
        bam = args.bam
        if args.cram:
            bam = convert_cram_to_bam(args.cram, temp_folder, args.reference_genome_fasta, bed_file)
        unsplit_2d_reads = detect_unsplit_2d_reads(bam, merged_loci, cfg.extension)
        if not bam_has_methylation(bam):
            logging.warning("The input alignments are missing 5mC and/or 5hmC base modification (MM) tags, so no methylation will be reported.")

        plot_dir = safe_output_path(args.out_dir, args.sample, "_plots") if (cfg.generate_clustering_plots or cfg.generate_motif_plots) else None
        debug_output_basename = str(plot_dir / args.sample) if cfg.generate_clustering_plots and plot_dir is not None else None
        tr_regions = process_regions(
            merged_loci,
            bam,
            args.no_clustering,
            args.sex,
            unsplit_2d_reads,
            cfg,
            pysam.FastaFile(args.reference_genome_fasta),
            debug_output_basename=debug_output_basename,
        )
        tr_records = [record for region in tr_regions for record in region.reads]

        if not is_haplotagged(tr_records, args.sex):
            if not args.bypass_haplotag_check:
                raise UserFacingError(
                    "The input BAM file does not appear to be haplotagged. Halting execution. Use --bypass-haplotag-check to force execution."
                )
            else:
                logging.info(
                    "The input BAM file does not appear to be haplotagged. Continuing execution because --bypass-haplotag-check was specified."
                )

        logging.info("Saving per read results to tsv")
        write_tsv(tr_regions, safe_output_path(args.out_dir, args.sample, ".tsv"))

        logging.info("Saving per haplotype results to vcf")
        min_read_support = cfg.min_read_support_per_allele
        min_region_coverage = cfg.min_region_coverage
        loci_to_output = unmerged_loci
        if not cfg.include_extra_trs:
            loci_to_output = filter_loci_for_output(loci_to_output)
        write_vcf(
            tr_regions,
            args.sample,
            loci_to_output,
            args.reference_genome_fasta,
            bam,
            safe_output_path(args.out_dir, args.sample, ".vcf"),
            min_read_support,
            min_region_coverage,
            args.sex,
            command=shlex.join(sys.argv),
        )

        logging.info("Saving tagged bam")
        write_tagged_bam(
            bam,
            merged_loci,
            safe_output_path(args.out_dir, args.sample, ".tagged.bam"),
            tr_regions,
            Path(temp_folder),
            unsplit_2d_reads,
        )

        if cfg.generate_motif_plots:
            assert plot_dir is not None
            logging.info("Saving motif visualizations to %s", plot_dir)
            from mahi_mahi.motif_visualization import write_motif_plot_pdfs

            plot_paths = write_motif_plot_pdfs(
                tr_regions,
                sample_name=args.sample,
                output_dir=plot_dir,
                options=replace(
                    cfg.motif_plot.to_render_options(),
                    methylation_state_confidence_threshold=cfg.methylation_state_confidence_threshold,
                ),
                include_methylation=cfg.generate_methylation_plots,
            )
            logging.info("Saved %d motif visualization PDF(s)", len(plot_paths))

    logging.info("Process completed successfully.")


if __name__ == "__main__":
    main()
