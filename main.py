
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI

import storage
from decision_engine import decide
from schemas import (
    PROFILE,
    validate_commit_envelope,
    validate_dossiers,
    validate_proposal,
    validate_top_level_propose,
)
from security import (
    canonical_digest,
    canonical_json,
    proposal_digest,
    verify_receipt_signature,
)
from storage import (
    Evaluation,
    SessionLocal,
    get_cached_proposal,
    list_proposals,
    save_cached_proposal,
    save_evaluation,
    save_receipts_and_commit_response,
)

app = FastAPI(
    title="Safe AI Mailroom",
    version="2.0.0",
)

MAX_REQUEST_BYTES = 5 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

client = (
    OpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)


@app.on_event("startup")
def startup():
    storage.init_db()


def json_error(status_code, message):
    return JSONResponse(
        status_code=status_code,
        content={"error": message},
    )


def response_json(data, status_code=200):
    body = canonical_json(data).encode("utf-8")

    if len(body) > MAX_RESPONSE_BYTES:
        return json_error(413, "Response exceeds 512 KiB")

    return JSONResponse(
        status_code=status_code,
        content=json.loads(body),
        media_type="application/json",
    )


async def read_request(request: Request):
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type != "application/json":
        raise ValueError(
            "Content-Type must be application/json"
        )

    content_length = request.headers.get("content-length")

    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                raise OverflowError("Request body is too large")
        except ValueError:
            raise ValueError("Invalid Content-Length")

    raw = await request.body()

    if len(raw) > MAX_REQUEST_BYTES:
        raise OverflowError("Request body is too large")

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Malformed JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")

    return data


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/model-test")
def model_test():
    if client is None:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": "Model client not configured",
            },
        )

    try:
        result = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with OK only.",
                }
            ],
            max_tokens=20,
            timeout=15,
        )

        return {
            "ok": True,
            "model": MODEL_NAME,
            "reply": result.choices[0].message.content,
        }

    except Exception as exc:
        print("Model test failed:", type(exc).__name__)

        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": type(exc).__name__,
            },
        )


REQUEST_BODY_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["operation"],
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["propose", "commit"],
                        }
                    },
                    "additionalProperties": True,
                }
            }
        },
    }
}


@app.post("/", openapi_extra=REQUEST_BODY_SCHEMA)
@app.post(
    "/v1/mailroom/actions",
    openapi_extra=REQUEST_BODY_SCHEMA,
)
async def mailroom(request: Request):
    try:
        data = await read_request(request)

    except OverflowError as exc:
        return json_error(413, str(exc))

    except ValueError as exc:
        return json_error(400, str(exc))

    operation = data.get("operation")

    if operation == "propose":
        return await handle_propose(data)

    if operation == "commit":
        return handle_commit(data)

    return json_error(422, "Unsupported operation")


async def handle_propose(data):
    try:
        validate_top_level_propose(data)

        dossiers = data["dossiers"]
        validate_dossiers(dossiers)

        input_digest = canonical_digest(dossiers)
        request_hash = canonical_digest(data)
        evaluation_id = data["evaluationId"]
        public_jwk = data["receiptVerifier"]["publicKeyJwk"]

    except (ValueError, TypeError, KeyError) as exc:
        return json_error(422, str(exc))

    # An exact replay must reuse the stored response.
    with SessionLocal() as session:
        existing = session.get(Evaluation, evaluation_id)

        if existing:
            if existing.request_hash != request_hash:
                return json_error(
                    409,
                    "Evaluation ID reused with changed content",
                )

            return response_json(
                json.loads(existing.propose_response)
            )

    proposals = []

    try:
        with SessionLocal() as session:
            # Build all proposals before saving the evaluation.
            for dossier in dossiers:
                fingerprint = canonical_digest(dossier)

                cached = get_cached_proposal(
                    session,
                    fingerprint,
                )

                if cached is None:
                    returned_fingerprint, proposal = decide(dossier)

                    # Use the fingerprint returned by the decision engine
                    # only if it matches the canonical dossier fingerprint.
                    if returned_fingerprint != fingerprint:
                        raise ValueError(
                            "Decision engine fingerprint mismatch"
                        )

                    validate_proposal(
                        proposal,
                        dossier,
                        data["allowedActions"],
                    )

                    save_cached_proposal(
                        session,
                        fingerprint,
                        proposal,
                    )

                else:
                    proposal = cached

                    # Cached output must always be revalidated.
                    validate_proposal(
                        proposal,
                        dossier,
                        data["allowedActions"],
                    )

                proposals.append({
                    "proposal": proposal,
                    "fingerprint": fingerprint,
                    "proposalDigest": proposal_digest(proposal),
                })

            response = {
                "profile": PROFILE,
                "evaluationId": evaluation_id,
                "status": "awaiting_receipts",
                "inputDigest": input_digest,
                "proposals": [
                    item["proposal"]
                    for item in proposals
                ],
            }

            if (
                len(canonical_json(response).encode("utf-8"))
                > MAX_RESPONSE_BYTES
            ):
                session.rollback()
                return json_error(
                    413,
                    "Response exceeds 512 KiB",
                )

            save_evaluation(
                session,
                evaluation_id,
                request_hash,
                input_digest,
                public_jwk,
                response,
                proposals,
            )

            session.commit()

        return response_json(response)

    except Exception as exc:
        print("Propose failed:", type(exc).__name__)

        return json_error(
            503,
            "Unable to safely process proposals",
        )


def handle_commit(data):
    try:
        validate_commit_envelope(data)

        evaluation_id = data["evaluationId"]
        input_digest = data["inputDigest"]
        receipts = data["receipts"]
        request_hash = canonical_digest(data)

    except (ValueError, TypeError, KeyError) as exc:
        return json_error(422, str(exc))

    try:
        with SessionLocal() as session:
            # Serialize commit processing so two concurrent commits
            # cannot both verify and save different receipt batches.
            evaluation = session.execute(
                select(Evaluation)
                .where(Evaluation.id == evaluation_id)
                .with_for_update()
            ).scalar_one_or_none()

            if evaluation is None:
                return json_error(409, "Unknown evaluation")

            if evaluation.input_digest != input_digest:
                return json_error(409, "Input digest mismatch")

            # Exact commit replay.
            if evaluation.commit_response is not None:
                if evaluation.commit_request_hash != request_hash:
                    return json_error(
                        409,
                        "Evaluation already committed with different content",
                    )

                return response_json(
                    json.loads(evaluation.commit_response)
                )

            stored = list_proposals(session, evaluation_id)

            if len(receipts) != len(stored):
                return json_error(
                    422,
                    "Receipt count does not match proposal count",
                )

            by_dossier = {
                item["dossierId"]: item
                for item in stored
            }

            if len(by_dossier) != len(stored):
                return json_error(
                    500,
                    "Stored proposal conflict",
                )

            seen_dossiers = set()
            seen_receipt_ids = set()
            verified = {}

            public_jwk = json.loads(evaluation.verifier_jwk)

            # Validate every receipt before persisting any receipt.
            for receipt in receipts:
                dossier_id = receipt["dossierId"]
                receipt_id = receipt["receiptId"]

                if dossier_id in seen_dossiers:
                    return json_error(
                        422,
                        "Duplicate dossier receipt",
                    )

                seen_dossiers.add(dossier_id)

                if receipt_id in seen_receipt_ids:
                    return json_error(
                        422,
                        "Duplicate receipt ID",
                    )

                seen_receipt_ids.add(receipt_id)

                saved = by_dossier.get(dossier_id)

                if saved is None:
                    return json_error(
                        422,
                        "Receipt for unknown dossier",
                    )

                proposal = saved["proposal"]

                if receipt["callId"] != proposal["callId"]:
                    return json_error(
                        422,
                        "Receipt callId mismatch",
                    )

                if receipt["action"] != proposal["action"]:
                    return json_error(
                        422,
                        "Receipt action mismatch",
                    )

                expected_digest = saved["proposalDigest"]

                if receipt["proposalDigest"] != expected_digest:
                    return json_error(
                        422,
                        "Proposal digest mismatch",
                    )

                verify_receipt_signature(
                    public_jwk,
                    evaluation_id,
                    input_digest,
                    receipt,
                )

                verified[dossier_id] = receipt

            if seen_dossiers != set(by_dossier):
                return json_error(
                    422,
                    "Missing proposal receipt",
                )

            # All receipt signatures and scopes have passed.
            outcomes = []

            for saved in stored:
                dossier_id = saved["dossierId"]
                receipt = verified[dossier_id]

                outcomes.append({
                    "dossierId": dossier_id,
                    "callId": saved["callId"],
                    "action": saved["proposal"]["action"],
                    "proposalDigest": saved["proposalDigest"],
                    "receiptId": receipt["receiptId"],
                    "status": (
                        "executed"
                        if receipt["accepted"]
                        else "rejected"
                    ),
                })

            response = save_receipts_and_commit_response(
                session,
                evaluation,
                [
                    verified[item["dossierId"]]
                    for item in stored
                ],
                outcomes,
                request_hash,
            )

            session.commit()

            return response_json(response)

    except ValueError as exc:
        return json_error(422, str(exc))

    except Exception as exc:
        print("Commit failed:", type(exc).__name__)

        return json_error(
            503,
            "Unable to safely verify commit",
        )
