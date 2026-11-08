"""Experiment matrix (spec §5.4) loaded from YAML configs bundled in ``legalrag/eval/configs``."""

from __future__ import annotations

import itertools
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from legalrag.retrieval.types import RetrievalOverrides

_FIELDS = ("strategy", "mode", "use_rerank", "use_rewrite", "top_k", "embedding_model")


class ExperimentConfig(BaseModel):
    name: str
    strategy: str = "article"
    mode: str = "hybrid"
    use_rerank: bool = True
    use_rewrite: bool = True
    top_k: int = 6
    embedding_model: str | None = None
    strategy_suffix: str = ""

    def overrides(self) -> RetrievalOverrides:
        return RetrievalOverrides(
            top_k=self.top_k,
            use_rerank=self.use_rerank,
            use_rewrite=self.use_rewrite,
            strategy=self.strategy + self.strategy_suffix,
            mode=self.mode,  # type: ignore[arg-type]
        )

    def params(self) -> dict[str, Any]:
        return self.model_dump(exclude={"name"})


def _configs_dir() -> Any:
    return resources.files("legalrag.eval").joinpath("configs")


def suite_names() -> list[str]:
    return sorted(p.name[:-5] for p in _configs_dir().iterdir() if p.name.endswith(".yaml"))


def _read_yaml(name_or_path: str) -> dict[str, Any]:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    res = _configs_dir().joinpath(f"{name_or_path}.yaml")
    if not res.is_file():
        raise FileNotFoundError(
            f"no experiment config '{name_or_path}' (available: {suite_names()})"
        )
    return yaml.safe_load(res.read_text(encoding="utf-8"))


def _expand(doc: dict[str, Any]) -> list[ExperimentConfig]:
    name = doc["name"]
    base = {k: v for k, v in (doc.get("base") or {}).items() if k in _FIELDS}
    suffix = doc.get("strategy_suffix", "")
    out: list[ExperimentConfig] = []
    if "variants" in doc:
        for v in doc["variants"]:
            params = {**base, **{k: val for k, val in v.items() if k in _FIELDS}}
            out.append(
                ExperimentConfig(name=f"{name}-{v['suffix']}", strategy_suffix=suffix, **params)
            )
    elif "grid" in doc:
        keys = list(doc["grid"])
        for combo in itertools.product(*(doc["grid"][k] for k in keys)):
            params = {**base, **dict(zip(keys, combo, strict=True))}
            label = "_".join(f"{k}_{v}" for k, v in zip(keys, combo, strict=True))
            out.append(ExperimentConfig(name=f"{name}-{label}", strategy_suffix=suffix, **params))
    else:
        out.append(ExperimentConfig(name=name, strategy_suffix=suffix, **base))
    return out


def load_suite(name_or_path: str, *, full: bool = False) -> Any:
    """Return the expanded configs of a matrix, or (``full=True``) a suite dict with
    ``retrieval``/``answer`` sections for composite suites such as ``default``."""
    doc = _read_yaml(name_or_path)
    if "retrieval" in doc or "answer" in doc:
        suite: dict[str, Any] = {}
        for part in ("retrieval", "answer"):
            if part not in doc:
                continue
            cfgs: list[ExperimentConfig] = []
            for ref in doc[part]["configs"]:
                matrix = ref.split("-", 1)[0]
                cfgs.extend(c for c in load_suite(matrix) if _matches(c, ref))
            suite[part] = {**doc[part], "configs": cfgs}
        if full:
            return suite
        return [c for part in suite.values() for c in part["configs"]]
    return _expand(doc)


def _matches(cfg: ExperimentConfig, ref: str) -> bool:
    """`E2` selects the whole matrix; `E2-hybrid_rerank` selects one expanded config."""
    base = ref.split("-", 1)[0]
    return cfg.name == ref or (ref == base and cfg.name.startswith(base))
