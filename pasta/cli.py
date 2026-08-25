#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from .pasta import PastaConfig, PastaSimulator


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def unit_float(value: str) -> float:
    fvalue = float(value)
    if not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value!r}")
    return fvalue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pasta",
        description=(
            "PASTA simulator: generate a synthetic dataset of subjective "
            "annotations, with independent control over ambiguity "
            "(task/stimulus difficulty) and subjectivity (annotator "
            "disagreement)."
        ),
        epilog=(
            "Example:\n"
            "  pasta --ambiguity 0.5 --subjectivity 0.5 -o annotations.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    main_group = parser.add_argument_group("main parameters")
    main_group.add_argument(
        "--ambiguity",
        type=unit_float,
        default=0.3,
        metavar="[0-1]",
        help=(
            "how ambiguous the annotation task is, from 0 (fully "
            "deterministic) to 1 (random noise). (default: %(default)s)"
        ),
    )
    main_group.add_argument(
        "--subjectivity",
        type=unit_float,
        default=0.3,
        metavar="[0-1]",
        help=(
            "how subjective the task is, from 0 (all annotators share one "
            "objective view) to 1 (labeling depends entirely on individual "
            "perception). (default: %(default)s)"
        ),
    )

    annot_group = parser.add_argument_group("annotation parameters")
    annot_group.add_argument(
        "--n-instances",
        type=positive_int,
        default=1000,
        metavar="N",
        help="number of instances to generate. (default: %(default)s)",
    )
    annot_group.add_argument(
        "--n-annotators",
        type=positive_int,
        default=10,
        metavar="N",
        help="number of simulated annotators. (default: %(default)s)",
    )
    annot_group.add_argument(
        "--n-classes",
        type=positive_int,
        default=2,
        metavar="N",
        help="number of possible labels. (default: %(default)s)",
    )
    annot_group.add_argument(
        "--embedding-dim",
        type=positive_int,
        default=64,
        metavar="N",
        help=(
            "number of dimensions of the instance/annotator embedding "
            "space. (default: %(default)s)"
        ),
    )
    annot_group.add_argument(
        "--n-annotator-subgroups",
        type=positive_int,
        default=2,
        metavar="N",
        help="how many groups the annotators are polarized into. (default: %(default)s)",
    )
    annot_group.add_argument(
        "--annotations-per-instance",
        type=positive_int,
        default=None,
        metavar="N",
        help=(
            "how many annotators should label each instance. "
            "(default: all annotators label every instance)"
        ),
    )
    annot_group.add_argument(
        "--repeats",
        type=positive_int,
        default=1,
        metavar="N",
        help="how many times each annotator repeats the labeling. (default: %(default)s)",
    )

    tweak_group = parser.add_argument_group("tweaks")
    tweak_group.add_argument(
        "--class-separation",
        type=float,
        default=6.0,
        metavar="F",
        help="distance between class prototypes. (default: %(default)s)",
    )
    tweak_group.add_argument(
        "--instance-spread",
        type=float,
        default=1.0,
        metavar="F",
        help=(
            "how tightly instances cluster around their class. "
            "(default: %(default)s)"
        ),
    )
    tweak_group.add_argument(
        "--annotator-subgroup-separation",
        type=float,
        default=6.0,
        metavar="F",
        help="distance between annotator groups' centers. (default: %(default)s)",
    )
    tweak_group.add_argument(
        "--annotator-spread",
        type=float,
        default=None,
        metavar="F",
        help=(
            "individual variation within a subgroup. "
            "(default: falls back to --class-separation)"
        ),
    )
    tweak_group.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="seed for the random number generator. (default: %(default)s)",
    )

    io_group = parser.add_argument_group("output")
    io_group.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="path to write the resulting CSV to. (default: print to stdout)",
    )
    io_group.add_argument(
        "--head",
        type=positive_int,
        default=None,
        metavar="N",
        help="only print/save the first N rows of the annotated dataset.",
    )
    io_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the summary message printed after writing a file.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = PastaConfig(
            n_instances=args.n_instances,
            n_annotators=args.n_annotators,
            n_classes=args.n_classes,
            embedding_dim=args.embedding_dim,
            subjectivity=args.subjectivity,
            ambiguity=args.ambiguity,
            class_separation=args.class_separation,
            instance_spread=args.instance_spread,
            n_annotator_subgroups=args.n_annotator_subgroups,
            annotator_subgroup_separation=args.annotator_subgroup_separation,
            annotator_spread=args.annotator_spread,
            annotations_per_instance=args.annotations_per_instance,
            repeats=args.repeats,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # unreachable, parser.error exits, but keeps type-checkers happy

    simulator = PastaSimulator(config)
    df = simulator.annotate()

    if args.head is not None:
        df = df.head(args.head)

    if args.output:
        df.to_csv(args.output, index=False)
        if not args.quiet:
            print(
                f"Wrote {len(df)} annotations "
                f"({args.n_instances} instances x {args.n_annotators} annotators) "
                f"to {args.output}",
                file=sys.stderr,
            )
    else:
        df.to_csv(sys.stdout, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())