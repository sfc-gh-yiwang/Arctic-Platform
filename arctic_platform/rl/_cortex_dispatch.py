# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cortex dispatch shim for the legacy ``arctic_platform.rl`` API.

SkyRL / verl adapters build ``arctic_platform.rl.ArcticRLClientConfig`` and
expect an async client. This file translates that legacy config into the
nested ``arctic_platform.client.ArcticRLClientConfig`` (with a Cortex
``backend_config``), then wraps the natively-async
``arctic_platform.client.ArcticRLClient`` behind the legacy method surface.

Cortex-specific bits still on this side:

* ``fwd_bwd`` payload reshape: SkyRL / verl push ``{batch, meta, processing}``;
  Cortex expects ``{args, kwargs, context, processing}``.
* ``fwd_no_grad`` returns ``[B, T]`` zeros — Cortex has no ``/forward`` and the
  server-side GRPO loss defaults ``old_log_probs = logprobs.detach()`` when
  ``context.old_log_probs_shifted`` is absent, i.e. π_old ≡ π_new, correct for
  single-epoch on-policy.
* Cortex-only stubs (no colocation lifecycle, no ``/log-probs``, no
  ``save_weights``).

All Cortex deployment settings come from ``ARCTIC_CORTEX_*`` env vars; the
legacy ``ArcticRLClientConfig`` intentionally carries none.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from arctic_platform.rl.config import ArcticRLClientConfig as _LegacyConfig

logger = logging.getLogger(__name__)


def _cortex_backend_config():
    from arctic_platform.client.config import CortexConfig

    return CortexConfig(
        base_url=os.environ.get("ARCTIC_CORTEX_BASE_URL"),
        host=os.environ.get("ARCTIC_CORTEX_HOST"),
        pat_env_var=os.environ.get("ARCTIC_CORTEX_PAT_ENV_VAR") or "CORTEX_PAT",
        database=os.environ.get("ARCTIC_CORTEX_DATABASE") or "",
        endpoint=os.environ.get("ARCTIC_CORTEX_ENDPOINT") or "cortex-training",
        **{"schema": os.environ.get("ARCTIC_CORTEX_SCHEMA") or ""},
    )


def _to_unified_config(legacy: _LegacyConfig):
    """Legacy ``arctic_platform.rl`` config -> nested unified client config.

    Optimizer + gradient-clipping are merged into ``training.ds_config`` so
    Cortex's ``to_cortex()`` sub-job builder can lift them; the legacy
    ``training_config`` dict is otherwise opaque to Cortex.
    """
    from arctic_platform.client.config import (
        ArcticRLClientConfig as U, SamplingConfig, TrainingConfig,
    )

    ds_config = dict(legacy.ds_config or {})
    tc_in = dict(legacy.training_config or {})
    if "optimizer" in tc_in and "optimizer" not in ds_config:
        ds_config["optimizer"] = tc_in["optimizer"]

    fields: dict[str, Any] = {
        "model_name": legacy.model_name,
        "seed": legacy.seed,
        "training_gpus": legacy.training_gpus,
        "sampling_gpus": legacy.sampling_gpus,
        "log_prob_gpus": legacy.log_prob_gpus,
        "job_ready_timeout": legacy.job_ready_timeout,
        "backend_config": _cortex_backend_config(),
        "training": TrainingConfig(
            ds_config=ds_config or None,
            ds_worker_config=legacy.ds_worker_config or None,
            checkpoint_path=legacy.checkpoint_path,
            full_determinism=legacy.full_determinism,
        ),
        "sampling": SamplingConfig(
            vllm=dict(legacy.vllm_config or {}),
            arctic_inference_config=legacy.arctic_inference_config or None,
        ),
    }
    for k in ("training_job_id", "sampling_job_id", "log_prob_job_id"):
        v = getattr(legacy, k, None)
        if v is not None:
            fields[k] = v
    return U(**fields)


class _CortexClientShim:
    """Async facade over ``arctic_platform.client.ArcticRLClient``.

    Every hot-path method delegates to the natively-async unified client.
    The remaining code is Cortex-specific request/stub translation.
    """

    def __init__(self, legacy_config: _LegacyConfig) -> None:
        from arctic_platform.client import ArcticRLClient

        self._legacy_config = legacy_config
        self._unified_config = _to_unified_config(legacy_config)
        self._client = ArcticRLClient(self._unified_config)

    # ── legacy surface expected by SkyRL / verl adapters ─────────────
    @property
    def config(self) -> _LegacyConfig: return self._legacy_config
    @property
    def training_job_id(self) -> Any: return self._client.jobs.training
    @property
    def sampling_job_id(self) -> Any: return self._client.jobs.sampling
    @property
    def log_prob_job_id(self) -> Any: return self._client.jobs.log_prob
    @property
    def server_state(self) -> Any: return None
    def get_server_state(self) -> Any: return None

    def reconnect_config(self) -> _LegacyConfig:
        return self._legacy_config.model_copy(update={
            "training_job_id": self._client.jobs.training,
            "sampling_job_id": self._client.jobs.sampling,
            "log_prob_job_id": self._client.jobs.log_prob,
        })

    def shutdown(self):
        # SkyRL calls this sync; verl awaits it. Inside a running loop return
        # the coroutine; otherwise drive it to completion in a fresh loop so
        # sync callers don't leak jobs.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._client.shutdown())
            return None
        return self._client.shutdown()

    # ── training ─────────────────────────────────────────────────────
    async def fwd_bwd(self, batch: dict, **legacy_kwargs: Any) -> dict:
        # SkyRL/verl ``{batch, meta, processing}`` -> Cortex canonical
        # ``{args, kwargs, context, processing}``. ``old_log_probs_shifted``
        # is intentionally omitted: server-side loss defaults it to
        # ``logprobs.detach()`` (π_old ≡ π_new), correct for single-epoch
        # on-policy.
        import torch

        payload = dict(batch)
        processing_in = legacy_kwargs.pop("processing", None) or payload.pop("processing", None)
        payload.pop("router_replay", None)
        legacy_kwargs.pop("router_replay", None)

        if "batch" in payload and isinstance(payload["batch"], dict):
            tensors, meta = dict(payload["batch"]), dict(payload.get("meta") or {})
        else:
            tensors = dict(payload)
            meta = dict(tensors.pop("context", None) or {})

        input_ids = tensors.get("input_ids")
        attention_mask = tensors.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise ValueError("cortex fwd_bwd requires 'input_ids' and 'attention_mask'")

        loss_mask = tensors.pop("loss_mask", None)
        if loss_mask is None:
            loss_mask = tensors.pop("response_mask", None)
        if loss_mask is None:
            loss_mask = attention_mask
        if torch.is_tensor(loss_mask):
            loss_mask = loss_mask.to(torch.bool)
        advantages = tensors.pop("advantages", None)
        if advantages is None:
            raise ValueError("cortex fwd_bwd requires 'advantages' [B, S]")
        tensors.pop("old_log_probs", None)

        kwargs_out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        for k in ("position_ids", "labels"):
            if k in tensors:
                kwargs_out[k] = tensors[k]

        proc_config = dict((processing_in or {}).get("config") or {})
        proc_config.setdefault("eps_clip", 0.2)
        proc_config.setdefault("prox_logp_method", "recompute")
        proc_config.setdefault("dp_size", int(self._unified_config.training_gpus or 1))
        for k in ("batch_num_tokens", "global_batch_size"):
            if k not in proc_config and k in meta:
                proc_config[k] = int(meta[k])

        return await self._client.fwd_bwd({
            "args": (), "kwargs": kwargs_out,
            "context": {"input_ids": input_ids, "advantages": advantages, "loss_mask": loss_mask},
            "processing": {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": proc_config},
        })

    async def fwd_no_grad(self, batch: dict, **_: Any) -> dict:
        import torch

        b_data = batch.get("batch") if isinstance(batch, dict) else None
        if not isinstance(b_data, dict):
            b_data = batch if isinstance(batch, dict) else {}
        ids = b_data.get("input_ids")
        b, t = (int(ids.shape[0]), int(ids.shape[-1])) if torch.is_tensor(ids) else (1, 1)
        z = torch.zeros((b, max(t, 1)), dtype=torch.float32)
        return {"batch": {"logprobs": z, "log_probs": z, "entropy": z, "entropies": z}}

    async def step(self, learning_rate: float | None = None) -> dict:
        return await self._client.step(learning_rate=learning_rate)

    async def save_checkpoint(self, stage_info: dict | None = None, path: str | None = None) -> dict:
        cid = stage_info.get("checkpoint_id") or stage_info.get("id") if isinstance(stage_info, dict) else None
        return await self._client.save_checkpoint(checkpoint_id=cid)

    async def save_weights(self, path: str) -> dict:
        logger.warning("save_weights is a no-op on Cortex (path=%s ignored)", path)
        return {}

    # ── sampling ─────────────────────────────────────────────────────
    async def generate(self, prompts, sampling_params=None, **kwargs) -> list:
        return await self._client.generate(
            prompts=prompts, sampling_params=sampling_params,
            routing_key=kwargs.pop("routing_key", None),
            strict=kwargs.pop("strict", False),
        )

    async def sync_weights(self, **_: Any) -> dict:
        return await self._client.sync_weights()

    async def reset_prefix_cache(self, drain: bool = True, timeout_s: float = 60.0) -> dict:
        return await self._client.reset_prefix_cache(drain=drain, timeout_s=timeout_s)

    # ── Cortex-only stubs ────────────────────────────────────────────
    async def wake_inference(self, **_): return {}
    async def sleep_inference(self, **_): return {}
    async def wake_training(self, **_): return {}
    async def sleep_training(self, **_): return {}
    async def wake_log_prob(self, **_): return {}
    async def sleep_log_prob(self, **_): return {}
    async def empty_training_cache(self, **_): return {}
    async def weight_norm(self, **_): return {}

    async def log_probs(self, batch: dict, **_) -> dict:
        raise NotImplementedError("Cortex has no log-probs endpoint; disable KL/ref-model recipes.")


def create_cortex_client(config: _LegacyConfig) -> _CortexClientShim:
    return _CortexClientShim(config)
