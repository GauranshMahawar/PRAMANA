"""Ed25519 signing -- the substrate, never the claim.

Sigstore ``model-transparency``, in-toto and SLSA bind (path, digest, signer identity)
and say nothing about behaviour. That project's own documentation is explicit that a
converted model has different bytes, so re-signing "only proves who produced those new
bytes". Attesting that a conversion *preserved behaviour* needs evidence the signature
format cannot carry, and supplying that evidence is D1's job, not this module's.

What this module is for: making every artefact, shard, manifest and ledger row
attributable to a named party, so that a finding can say *whose*.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from pramana.common.errors import SignatureInvalid


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


@dataclass
class Signer:
    """A named Ed25519 identity.

    ``signer_id`` is not cosmetic: it is the field a finding uses to name a party, and
    it is what makes `converter_uncertified` distinguishable from "we did not look".
    """

    signer_id: str
    _private: Ed25519PrivateKey

    @classmethod
    def generate(cls, signer_id: str) -> Signer:
        return cls(signer_id=signer_id, _private=Ed25519PrivateKey.generate())

    @classmethod
    def from_seed(cls, signer_id: str, seed: bytes) -> Signer:
        """Deterministic key from a 32-byte seed.

        For fixtures and reproducible demos only. A deployed instance generates its key
        in a ceremony and never derives it from anything that appears in a repository.
        """
        if len(seed) != 32:
            raise ValueError("Ed25519 seed must be exactly 32 bytes.")
        return cls(signer_id=signer_id, _private=Ed25519PrivateKey.from_private_bytes(seed))

    @classmethod
    def load(cls, signer_id: str, path: str | Path, password: bytes | None = None) -> Signer:
        data = Path(path).read_bytes()
        key = serialization.load_pem_private_key(data, password=password)
        if not isinstance(key, Ed25519PrivateKey):
            raise SignatureInvalid(f"{path} does not hold an Ed25519 private key.")
        return cls(signer_id=signer_id, _private=key)

    def save(self, path: str | Path, password: bytes | None = None) -> None:
        enc = (
            serialization.BestAvailableEncryption(password)
            if password
            else serialization.NoEncryption()
        )
        Path(path).write_bytes(
            self._private.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=enc,
            )
        )

    def sign(self, message: bytes) -> str:
        return _b64(self._private.sign(message))

    @property
    def public_key_bytes(self) -> bytes:
        return self._private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def public_key_b64(self) -> str:
        return _b64(self.public_key_bytes)


def verify(public_key: bytes | str, message: bytes, signature: str) -> bool:
    """Verify an Ed25519 signature. Returns False rather than raising on mismatch.

    A verification failure is an expected outcome in this system -- demo beat 12 is
    literally a verification failing on purpose -- so it is a value, not an exception.
    """
    raw = _unb64(public_key) if isinstance(public_key, str) else public_key
    try:
        Ed25519PublicKey.from_public_bytes(raw).verify(_unb64(signature), message)
        return True
    except (InvalidSignature, ValueError):
        return False


class KeyRegistry:
    """Maps ``signer_id`` to a public key.

    Deliberately plain. Key distribution, rotation and revocation are a PKI problem,
    declared out of scope in ``taxonomy.yaml`` as ``contributor_key_compromise``, and
    pretending otherwise here would be the kind of quiet overreach the project's own
    audit passes exist to catch.
    """

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}

    def register(self, signer_id: str, public_key: bytes | str) -> None:
        raw = _unb64(public_key) if isinstance(public_key, str) else public_key
        if signer_id in self._keys and self._keys[signer_id] != raw:
            raise SignatureInvalid(
                f"Refusing to silently rebind signer {signer_id!r} to a different key. "
                f"A signer identity that changes key without ceremony is how an "
                f"attribution becomes unfalsifiable."
            )
        self._keys[signer_id] = raw

    def register_signer(self, signer: Signer) -> None:
        self.register(signer.signer_id, signer.public_key_bytes)

    def get(self, signer_id: str) -> bytes | None:
        return self._keys.get(signer_id)

    def as_dict(self) -> dict[str, bytes]:
        return dict(self._keys)

    def __contains__(self, signer_id: object) -> bool:
        return signer_id in self._keys
