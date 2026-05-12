from __future__ import annotations

import typer

from refract.datasets.constants import ALL_DATASETS
from refract.datasets.real_world import load_real_world_dataset

app = typer.Typer(pretty_exceptions_enable=False)

_CLASSIC = [d for d in ALL_DATASETS if d not in {"roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"}]
_HETEROPHILIC = ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]


@app.command()
def run(
    data_dir: str = typer.Option("data", help="Root directory for dataset storage"),
    datasets: str = typer.Option("all", help="all | heterophilic | classic | comma-separated names"),
) -> None:
    if datasets == "all":
        names = list(ALL_DATASETS)
    elif datasets == "heterophilic":
        names = _HETEROPHILIC
    elif datasets == "classic":
        names = _CLASSIC
    else:
        names = [d.strip() for d in datasets.split(",")]

    typer.echo(f"Downloading {len(names)} dataset(s) → {data_dir}\n")
    typer.echo(f"  {'Dataset':<20} {'Nodes':>8} {'Edges':>10} {'Classes':>8} {'Features':>10}")
    typer.echo(f"  {'-'*58}")

    for name in names:
        try:
            ds = load_real_world_dataset(name, root=data_dir, split_idx=0)
            typer.echo(
                f"  {name:<20} {ds.num_nodes:>8,} {ds.num_edges:>10,} "
                f"{ds.num_classes:>8} {ds.num_features:>10}"
            )
        except Exception as e:
            typer.echo(f"  {name:<20} FAILED: {e}", err=True)

    typer.echo(f"\nDone. Data stored in ./{data_dir}/")


if __name__ == "__main__":
    app()
