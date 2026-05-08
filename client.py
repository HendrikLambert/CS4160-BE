from __future__ import annotations
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from ipv8.community import Community, CommunitySettings
from ipv8.configuration import (
    ConfigBuilder,
    Strategy,
    WalkerDefinition,
    default_bootstrap_defs,
)
from ipv8.lazy_community import Peer, lazy_wrapper
from ipv8.messaging.lazy_payload import VariablePayload, vp_compile
from ipv8_service import IPv8
from miner import DIFFICULTY_BITS, mine

COMMUNITY_ID = bytes.fromhex("2c1cc6e35ff484f99ebdfb6108477783c0102881")
SERVER_PUBKEY = bytes.fromhex(
    "4c69624e61434c504b3a86b23934a28d669c390e2d1fc0b0870706c4591cc0cb178bc5a811da6d87d27ef319b2638ef60cc8d119724f4c53a1ebfad919c3ac4136c501ce5c09364e0ebb"
)


class LogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not record.exc_info:
            return True
        exception = record.exc_info[1]
        is_legacy_curve = "1.3.132.0.1" in str(exception)
        return not is_legacy_curve


logging.getLogger("BECommunity").addFilter(LogFilter())


# dataclass type gave weird erros, so switched to VP
# https://py-ipv8.readthedocs.io/en/latest/reference/serialization.html
@vp_compile
class SubmissionPayload(VariablePayload):
    msg_id = 1
    format_list = ["varlenHutf8", "varlenHutf8", "q"]
    names = ["email", "github_url", "nonce"]


@vp_compile
class ResponsePayload(VariablePayload):
    msg_id = 2
    format_list = ["?", "varlenHutf8"]
    names = ["success", "message"]


class BECommunity(Community):
    community_id = COMMUNITY_ID

    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)
        self.add_message_handler(ResponsePayload, self.on_response)
        self.email: str = ""
        self.github_url: str = ""
        self.nonce: int = 0
        self.response_future: asyncio.Future | None = None
        self._submitted_to: set[bytes] = set()

    def configure(self, email: str, github_url: str, nonce: int) -> None:
        self.email = email
        self.github_url = github_url
        self.nonce = nonce
        self.response_future = asyncio.get_running_loop().create_future()
        self.register_task("submit_loop", self._submit_loop, interval=3.0, delay=1.0)

    async def _submit_loop(self) -> None:
        if self.response_future is None or self.response_future.done():
            return
        for peer in self.get_peers():
            pk = peer.public_key.key_to_bin()
            if pk == SERVER_PUBKEY and peer.mid not in self._submitted_to:
                print(
                    f"Found server peer {peer.address}, submitting nonce {self.nonce}"
                )
                self.ez_send(
                    peer,
                    SubmissionPayload(self.email, self.github_url, self.nonce),
                )
                self._submitted_to.add(peer.mid)

    def on_peer_added(self, peer: Peer) -> None:
        print("I am:", self.my_peer, "I found:", peer)

    def on_peer_removed(self, peer: Peer) -> None:
        print("I am:", self.my_peer, "I lost:", peer)

    @lazy_wrapper(ResponsePayload)
    def on_response(self, peer, payload: ResponsePayload) -> None:
        if peer.public_key.key_to_bin() != SERVER_PUBKEY:
            print(f"[!] Ignoring response from non-server peer {peer.mid.hex()}")
            return
        verdict = "ACCEPTED" if payload.success else "REJECTED"
        print(f"{verdict}: {payload.message}")
        if self.response_future is not None and not self.response_future.done():
            self.response_future.set_result((payload.success, payload.message))


async def run_ipv8(
    key_path: Path,
    email: str,
    github_url: str,
    nonce: int,
    timeout: float,
) -> int:
    builder = ConfigBuilder().clear_keys().clear_overlays()
    builder.add_key("client", "curve25519", str(key_path))
    builder.add_overlay(
        "BECommunity",
        "client",
        [WalkerDefinition(Strategy.RandomWalk, 10, {"timeout": 3.0})],
        default_bootstrap_defs,
        {},
        [],
    )
    ipv8 = IPv8(
        builder.finalize(),
        extra_communities={"BECommunity": BECommunity},
    )
    await ipv8.start()
    overlay = ipv8.get_overlay(BECommunity)
    print(f"Public key: {overlay.my_peer.public_key.key_to_bin().hex()}")
    print(f"Peer mid: {overlay.my_peer.mid.hex()}")
    print(f"Server pubkey: {SERVER_PUBKEY.hex()}")
    print("Joining community and waiting for the server to be discovered...")
    overlay.configure(email, github_url, nonce)

    try:
        success, _msg = await asyncio.wait_for(
            overlay.response_future,
            timeout=timeout,
        )

        if success:
            return 0

    except asyncio.TimeoutError:
        print(f"No response after {timeout:.0f}s. ")
    finally:
        await ipv8.stop()

    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CS4160 client (assignment 1)")
    p.add_argument(
        "--email",
        required=True,
        help="TU Delft email",
    )
    p.add_argument(
        "--github-url",
        required=True,
        help="URL of your GitHub repo",
    )
    p.add_argument(
        "--key",
        default="key.pem",
        help="Path to IPv8 private key file (default: key.pem)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of mining workers (default: 4)",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU miner (default: False, CPU miner)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for the server response (default: 300)",
    )
    p.add_argument(
        "--mine-only",
        action="store_true",
        default=False,
        help="Mine the PoW and exit without contacting the server",
    )
    return p.parse_args()


def validate_inputs(email: str, github_url: str) -> str | None:
    if not (email.endswith("@tudelft.nl") or email.endswith("@student.tudelft.nl")):
        return f"Email must end in @tudelft.nl or @student.tudelft.nl, got: {email}"
    if len(email.encode("utf-8")) > 254 or "\n" in email:
        return "Email must be lt 254 bytes and contain no newline"
    if not github_url:
        return "GitHub URL must be non-empty"
    if len(github_url) > 512:
        return "GitHub URL must be lt 512 bytes"
    if any(c.isspace() or ord(c) < 0x20 for c in github_url):
        return "GitHub URL not valid."
    return None


def main() -> int:
    args = parse_args()
    err = validate_inputs(args.email, args.github_url)
    if err is not None:
        print(err)
        return 1

    print(f"Mining PoW (difficulty: {DIFFICULTY_BITS} bits)")
    time_start = time.time()
    nonce, digest_hex = mine(args.email, args.github_url, args.workers, args.gpu)
    time_elapsed = time.time() - time_start
    print(f"Found nonce={nonce}  hash={digest_hex}  in {time_elapsed:.1f}s")

    if args.mine_only:
        return 0

    return asyncio.run(
        run_ipv8(Path(args.key), args.email, args.github_url, nonce, args.timeout)
    )


if __name__ == "__main__":
    sys.exit(main())
