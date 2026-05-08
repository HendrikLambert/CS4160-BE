"""Multiprocess SHA-256 PoW miner for CS4160 Lab 1."""

from __future__ import annotations

import ctypes
import hashlib
import multiprocessing as mp
from pathlib import Path

DEFAULT_DIFFICULTY_BITS = 28
_CUDA_LIB_PATH = Path(__file__).parent / "libminer.so"


def meets_difficulty(digest: bytes, difficulty_bits: int) -> bool:
    return int.from_bytes(digest, "big") < (1 << (256 - difficulty_bits))


def compute_hash(email: str, github_url: str, nonce: int) -> bytes:
    payload = (
        email.encode("utf-8")
        + b"\n"
        + github_url.encode("utf-8")
        + b"\n"
        + nonce.to_bytes(8, "big")
    )
    return hashlib.sha256(payload).digest()


def _worker(
    prefix: bytes,
    worker_index: int,
    stride: int,
    target: int,
    found,
    result_q,
) -> None:
    base = hashlib.sha256(prefix)
    nonce = worker_index
    batch_size = 200_000
    while not found.is_set():
        for _ in range(batch_size):
            h = base.copy()
            h.update(nonce.to_bytes(8, "big"))
            digest = h.digest()
            if int.from_bytes(digest, "big") < target:
                result_q.put((nonce, digest.hex()))
                found.set()
                return
            nonce += stride


def _mine_gpu(prefix: bytes, difficulty_bits: int) -> tuple[int, str]:
    if not _CUDA_LIB_PATH.exists():
        raise FileNotFoundError(
            f"{_CUDA_LIB_PATH} not found. Build it with:\n"
            "    nvcc -O3 -arch=native -shared -Xcompiler -fPIC kernel.cu -o libminer.so"
        )
    lib = ctypes.CDLL(str(_CUDA_LIB_PATH))
    lib.cuda_sha256_pow.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),  # const BYTE *prefix
        ctypes.c_uint,  # WORD prefix_len
        ctypes.c_uint,  # WORD difficulty_bits
        ctypes.c_ulonglong,  # LONG nonce_start
        ctypes.POINTER(ctypes.c_ulonglong),  # LONG *out_nonce
        ctypes.POINTER(ctypes.c_ubyte),  # BYTE *out_hash (32 bytes)
    ]
    lib.cuda_sha256_pow.restype = None

    prefix_buf = (ctypes.c_ubyte * len(prefix)).from_buffer_copy(prefix)
    out_hash = (ctypes.c_ubyte * 32)()
    out_nonce = ctypes.c_ulonglong(0)

    lib.cuda_sha256_pow(
        prefix_buf,
        len(prefix),
        difficulty_bits,
        0,
        ctypes.byref(out_nonce),
        out_hash,
    )
    return int(out_nonce.value), bytes(out_hash).hex()


def _mine_cpu(prefix: bytes, n_workers: int, difficulty_bits: int) -> tuple[int, str]:
    target = 1 << (256 - difficulty_bits)
    ctx = mp.get_context("spawn")
    found = ctx.Event()
    result_q: mp.Queue[tuple[int, str]] = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker,
            args=(prefix, i, n_workers, target, found, result_q),
            daemon=True,
        )
        for i in range(n_workers)
    ]

    for p in procs:
        p.start()

    try:
        nonce, digest_hex = result_q.get()
        return nonce, digest_hex
    finally:
        found.set()
        for p in procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()


def mine(
    email: str,
    github_url: str,
    n_workers: int,
    on_gpu: bool,
    difficulty_bits: int = DEFAULT_DIFFICULTY_BITS,
) -> tuple[int, str]:
    prefix = email.encode("utf-8") + b"\n" + github_url.encode("utf-8") + b"\n"

    if on_gpu:
        nonce, digest_hex = _mine_gpu(prefix, difficulty_bits)
    else:
        nonce, digest_hex = _mine_cpu(prefix, n_workers, difficulty_bits)

    verify = compute_hash(email, github_url, nonce).hex()
    if verify != digest_hex or not meets_difficulty(bytes.fromhex(verify), difficulty_bits):
        raise RuntimeError(f"miner produced invalid solution: {nonce} {digest_hex}")
    return nonce, digest_hex
