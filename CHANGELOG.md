# Changelog

All notable changes to this project will be documented in this file.

Public release changes are listed here from v0.6.0 onward.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v1.0.0]

### Added
- Report CpG methylation states inside tandem repeats, with optional plots.
- Show selected excluded reads in TSV output and motif plots, with the reason for exclusion.
- Add --overwrite to allow output in a non-empty directory.
- Add a setting to control clustering plots.

### Changed
- Generate clustering and motif plots by default.
- Include all sub-repeats in VCF output by default. Set include_extra_trs=false to include only the main sub-repeat.
- Mark uncertain allele assignments as AMBIGUOUS in more cases, including apparent homozygous calls with many non-spanning reads.
- Improve the detection and reporting of unexpected motifs.
- Include the full command in the VCF header.
- Require --sample to be a filename stem and require an empty output directory unless --overwrite is set.

### Fixed
- Faster motif-plot rendering, particularly for long expansions.
- Correct VCF output for zero-length alleles and the reference header.
- Count CNBP CAGC interruptions as one motif.

### Removed
- Remove the SVTYPE=STR VCF INFO field for VCF 4.3 compatibility.

## [v0.7.1]

### Added
- Add CLSTATE to VCF output and clstate to TSV output. AMBIGUOUS means that the allele assignment may be wrong.

### Removed
- Remove the VCF CL field. Use CLSTATE instead.

## [v0.7.0]

### Added
- Add 23 hg38 loci from STRchive.
- Warn when many reads at a locus cannot be used because of secondary alignments or processing failures.

### Changed
- Improve detection of alleles with similar lengths.
- Improve detection of very long alleles with many non-spanning reads.
- Reduce use of haplotags, as they can be wrong. (They are still required in some cases, so we still recommend providing them.)
- Improve allele assignment for composite tandem repeats.
- Give clearer errors for invalid configuration, loci, alignment inputs, and paths.
- Improve motif analysis and plots for composite tandem repeats.
- Reduce the default region merge distance from 100 to 50.

### Removed
- Remove NOTCH2NLC from the default loci set because its mapping is unreliable.

## [v0.6.0]

### Added
- Report counts of common repeat interruptions.
- Add optional motif composition plots and more clustering output in debug mode.

### Changed
- Rename the commands to mahi-mahi and mahi-mahi-loci-tools, and rename the built-in STR loci BED files.
- Replace JSON configuration with validated YAML configuration.
- Improve the use of non-spanning reads in allele detection and assignment.
- Allow close allele length peaks when they are clearly separated, and reject very unequal peaks.
- Improve strand bias analysis and composite tandem repeat detection.
