
import base64
import hashlib
import json
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_digest(value):
    return sha256_hex(canonical_json(value))


def proposal_view(proposal):
    return {
        "dossierId": proposal["dossierId"],
        "callId": proposal["callId"],
        "action": proposal["action"],
        "target": proposal.get("target"),
        "payload": proposal["payload"],
        "evidence": sorted(proposal["evidence"]),
    }


def proposal_digest(proposal):
    return canonical_digest(proposal_view(proposal))


def decode_base64url(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Invalid base64url value")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid base64url characters")

    padded = value + "=" * (-len(value) % 4)
    result = base64.urlsafe_b64decode(padded)

    return result


def verify_receipt_signature(public_jwk, evaluation_id,
                             input_digest, receipt):
    if not isinstance(public_jwk, dict):
        raise ValueError("Missing public key")

    if public_jwk.get("kty") != "OKP":
        raise ValueError("Invalid key type")

    if public_jwk.get("crv") != "Ed25519":
        raise ValueError("Invalid key curve")

    public_bytes = decode_base64url(public_jwk.get("x", ""))

    if len(public_bytes) != 32:
        raise ValueError("Invalid Ed25519 public key length")

    signature_text = receipt.get("receiptSignature")

    if not isinstance(signature_text, str):
        raise ValueError("Missing receipt signature")

    # Signatures use standard base64, not base64url.
        try:
            signature = base64.b64decode(
             signature_text,
             validate=True,
            )
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(
            "Invalid receipt signature encoding"
        ) from exc
    # The signed receipt excludes receiptSignature.
    inner_receipt = {
        key: value
        for key, value in receipt.items()
        if key != "receiptSignature"
    }

    signed_object = {
        "profile": "ga5-mailroom-action-gate/v2",
        "evaluationId": evaluation_id,
        "inputDigest": input_digest,
        "receipt": inner_receipt,
    }

    message = canonical_json(signed_object).encode("utf-8")

    key = Ed25519PublicKey.from_public_bytes(public_bytes)

    try:
        key.verify(signature, message)
    except InvalidSignature as exc:
        raise ValueError("Invalid receipt signature") from exc

    return True


def make_call_id(dossier_id, fingerprint):
    # Stable across evaluations for identical dossier content.
    raw = sha256_hex(
        dossier_id + ":" + fingerprint
    )
    return "mailroom-" + raw[:48]


def valid_digest(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )
