"""
Benchmark: Categorical Sampling in JAX

Compares three implementations of sampling from a categorical distribution:
- jax_categorical        : JAX built-in (Gumbel-max internally) - sanity check
- gumbel_max_categorical : manual Gumbel-max trick
- cumsum_categorical     : softmax -> cumsum -> searchsorted (baseline)

Run from repo root:  python -m scripts.categorical_sampling
"""
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

from jaxfromscratch.sampling import (
    cumsum_categorical,
    gumbel_max_categorical,
    jax_categorical,
)


VOCAB_SIZES: tuple[int, ...] = (1024, 8192, 32768, 131072)
BATCH_SIZES: tuple[int, ...] = (1, 8, 64, 512)
WARMUP_ITERS = 20
MEASURE_ITERS = 200
SAMPLES_FOR_VALIDATION = 100_000
VALIDATION_VOCAB = 8
TV_THRESHOLD = 1e-3
OUTPUT_DIR = Path("scripts/results")


@dataclass(frozen=True)
class SamplerSpec:
    name: str
    sample: Callable[[PRNGKeyArray, Float[Array, "... vocab"]], Int[Array, "..."]]


@jaxtyped(typechecker=typechecker)
def validate(sampler, key: PRNGKeyArray, name: str) -> float:
    logits = jnp.asarray([0.0, 1.0, 2.0, 3.0, -1.0, 0.5, -0.5, 2.5])
    expected = jax.nn.softmax(logits)
    subkeys = jax.random.split(key, SAMPLES_FOR_VALIDATION)
    samples = jax.vmap(lambda k: sampler(k, logits))(subkeys)
    counts = jnp.bincount(samples, length=VALIDATION_VOCAB).astype(jnp.float32)
    empirical = counts / SAMPLES_FOR_VALIDATION
    tv = 0.5 * jnp.sum(jnp.abs(empirical - expected))
    print(f"  [{name}] total variation distance: {float(tv):.2e} (threshold {TV_THRESHOLD:.0e})")
    return float(tv)


@jaxtyped(typechecker=typechecker)
def time_one(sampler, logits: Float[Array, "batch vocab"], key: PRNGKeyArray) -> float:
    for _ in range(WARMUP_ITERS):
        sampler(key, logits).block_until_ready()
    times = []
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        sampler(key, logits).block_until_ready()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000.0


def build_samplers() -> list[SamplerSpec]:
    return [
        SamplerSpec(name="jax_random_categorical", sample=eqx.filter_jit(jax_categorical)),
        SamplerSpec(name="gumbel_max",             sample=eqx.filter_jit(gumbel_max_categorical)),
        SamplerSpec(name="softmax_cumsum",         sample=eqx.filter_jit(cumsum_categorical)),
    ]


def make_logits(vocab: int, batch: int, key: PRNGKeyArray) -> Float[Array, "batch vocab"]:
    return jax.random.normal(key, (batch, vocab))


def run_validation(samplers: list[SamplerSpec], key: PRNGKeyArray) -> bool:
    print("\n=== validation (Total Variation distance to softmax(logits)) ===")
    all_ok = True
    for spec in samplers:
        key, sub = jax.random.split(key)
        tv = validate(spec.sample, sub, spec.name)
        if tv > TV_THRESHOLD:
            print(f"  FAIL: {spec.name} has TV={tv:.2e}, exceeds {TV_THRESHOLD:.0e}")
            all_ok = False
    return all_ok


def run_benchmark(samplers: list[SamplerSpec], key: PRNGKeyArray) -> list[dict]:
    print("\n=== benchmark (median over {} iterations) ===".format(MEASURE_ITERS))
    rows: list[dict] = []
    for vocab in VOCAB_SIZES:
        for batch in BATCH_SIZES:
            key, sub = jax.random.split(key)
            logits = make_logits(vocab, batch, sub)
            row: dict = {"vocab": vocab, "batch": batch}
            for spec in samplers:
                t_ms = time_one(spec.sample, logits, sub)
                row[spec.name] = t_ms
            rows.append(row)
            print(
                f"  vocab={vocab:>6} batch={batch:>4} | "
                + " | ".join(f"{s.name}={row[s.name]:.3f}ms" for s in samplers)
            )
    return rows


def print_table(rows: list[dict], samplers: list[SamplerSpec]) -> None:
    if not rows:
        return
    cumsum_name = "softmax_cumsum"
    cols = ["vocab", "batch"] + [s.name for s in samplers] + [
        f"speedup_{s.name}_vs_{cumsum_name}" for s in samplers[:-1]
    ]
    widths = [
        max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) + 2
        for c in cols
    ]
    widths[0] = max(widths[0], 6)
    widths[1] = max(widths[1], 6)

    def fmt_row(cells):
        return "│" + "│".join(str(c).center(w) for c, w in zip(cells, widths)) + "│"

    sep = lambda l, m, r: l + m.join("─" * w for w in widths) + r

    print("\n=== summary ===")
    print(sep("┌", "┬", "┐"))
    print(fmt_row(cols))
    print(sep("├", "┼", "┤"))
    for r in rows:
        cells = [r["vocab"], r["batch"]]
        for s in samplers:
            cells.append(f"{r[s.name]:.3f} ms")
        for s in samplers[:-1]:
            cells.append(f"{r[cumsum_name] / r[s.name]:.2f}x")
        print(fmt_row(cells))
    print(sep("└", "┴", "┘"))


def save_csv(rows: list[dict], samplers: list[SamplerSpec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cumsum_name = "softmax_cumsum"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["vocab", "batch"] + [s.name for s in samplers] + [
            f"speedup_{s.name}_vs_{cumsum_name}" for s in samplers[:-1]
        ]
        writer.writerow(header)
        for r in rows:
            row = [r["vocab"], r["batch"]]
            row += [f"{r[s.name]:.6f}" for s in samplers]
            row += [f"{r[cumsum_name] / r[s.name]:.4f}" for s in samplers[:-1]]
            writer.writerow(row)
    print(f"\nSaved results -> {path}")


def main() -> None:
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices:     {jax.devices()}")

    samplers = build_samplers()
    key = jax.random.key(0)

    if not run_validation(samplers, key):
        print("\nValidation failed - skipping benchmark.")
        return

    key, sub = jax.random.split(key)
    rows = run_benchmark(samplers, sub)
    print_table(rows, samplers)
    save_csv(rows, samplers, OUTPUT_DIR / "categorical_sampling.csv")


if __name__ == "__main__":
    main()
