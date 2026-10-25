
import re

PROFILE = "ga5-mailroom-action-gate/v2"

ACTIONS = {
    "create_draft",
    "update_internal_record",
    "send_approved_notice",
    "request_confirmation",
    "quarantine_item",
    "no_action",
}

PAYLOAD_KEYS = {
    "create_draft": {
        "recipient", "referenceId", "status", "template"
    },
    "update_internal_record": {
        "field", "sourceEventId", "value"
    },
    "send_approved_notice": {
        "referenceId", "status", "template"
    },
    "request_confirmation": {
        "claimedSender", "questionCode", "referenceId"
    },
    "quarantine_item": {
        "artifactId", "reasonCode"
    },
    "no_action": {
        "reasonCode", "referenceId"
    },
}

TARGET_KINDS = {
    "create_draft": "draft_queue",
    "update_internal_record": "case_record",
    "send_approved_notice": "email",
    "request_confirmation": "approval_queue",
    "quarantine_item": "security_queue",
    "no_action": None,
}

REQUIRED_STRINGS = {
    "recipient",
    "referenceId",
    "status",
    "template",
    "field",
    "sourceEventId",
    "value",
    "claimedSender",
    "questionCode",
    "artifactId",
    "reasonCode",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_string(value):
    return isinstance(value, str) and bool(value.strip())


def validate_dossiers(dossiers):
    require(isinstance(dossiers, list), "dossiers must be a list")
    require(len(dossiers) > 0, "dossiers cannot be empty")

    seen_dossiers = set()

    for dossier in dossiers:
        require(isinstance(dossier, dict), "Invalid dossier")
        require(
            {"dossierId", "partition", "receivedAt", "mailbox",
             "objective", "sources"} <= dossier.keys(),
            "Missing dossier fields",
        )

        dossier_id = dossier["dossierId"]
        require(is_string(dossier_id), "Invalid dossierId")
        require(dossier_id not in seen_dossiers, "Duplicate dossierId")
        seen_dossiers.add(dossier_id)

        require(
            dossier["partition"] in {"stable_core", "fresh_audit"},
            "Invalid partition",
        )
        require(is_string(dossier["receivedAt"]), "Invalid receivedAt")
        require(is_string(dossier["mailbox"]), "Invalid mailbox")
        require(is_string(dossier["objective"]), "Invalid objective")
        require(isinstance(dossier["sources"], list), "Invalid sources")

        source_ids = set()
        line_ids = set()

        for source in dossier["sources"]:
            require(isinstance(source, dict), "Invalid source")
            for key in ("sourceId", "kind", "provenance", "title", "lines"):
                require(key in source, "Missing source field: " + key)

            require(is_string(source["sourceId"]), "Invalid sourceId")
            require(source["sourceId"] not in source_ids,
                    "Duplicate sourceId")
            source_ids.add(source["sourceId"])

            for key in ("kind", "provenance", "title"):
                require(is_string(source[key]), "Invalid source " + key)

            require(isinstance(source["lines"], list), "Invalid lines")

            for line in source["lines"]:
                require(isinstance(line, dict), "Invalid line")
                require(
                    is_string(line.get("lineId")),
                    "Invalid lineId",
                )
                require(
                    isinstance(line.get("text"), str),
                    "Invalid line text",
                )
                require(
                    line["lineId"] not in line_ids,
                    "Duplicate lineId",
                )
                line_ids.add(line["lineId"])


def validate_proposal(proposal, dossier, allowed_actions):
    require(isinstance(proposal, dict), "Proposal must be an object")

    expected_keys = {
        "dossierId", "callId", "action",
        "target", "payload", "evidence",
    }
    require(set(proposal) == expected_keys, "Invalid proposal fields")
    require(proposal["dossierId"] == dossier["dossierId"],
            "Wrong dossierId")
    require(
        isinstance(proposal["callId"], str)
        and re.fullmatch(r"[A-Za-z0-9._:-]{12,128}",
                         proposal["callId"]) is not None,
        "Invalid callId",
    )

    action = proposal["action"]
    require(action in ACTIONS, "Unknown action")
    require(action in allowed_actions, "Action not allowed")

    target = proposal["target"]
    payload = proposal["payload"]

    require(isinstance(payload, dict), "Payload must be an object")
    require(set(payload) == PAYLOAD_KEYS[action],
            "Incorrect payload fields")

    for key, value in payload.items():
        if key in REQUIRED_STRINGS:
            require(is_string(value), "Invalid payload value: " + key)

    if action == "create_draft":
        require(
            payload["template"] == "order_status",
            "Invalid draft template",
        )
        require(
            target == {
                "kind": "draft_queue",
                "id": "mailbox:" + dossier["mailbox"],
            },
            "Invalid draft target",
        )

    elif action == "update_internal_record":
        require(
            isinstance(target, dict)
            and set(target) == {"kind", "id"}
            and target["kind"] == "case_record"
            and is_string(target["id"]),
            "Invalid case record target",
        )
        require(payload["field"] == "delivery_window",
                "Invalid update field")

    elif action == "send_approved_notice":
        require(payload["template"] == "approved_delivery_notice",
                "Invalid notice template")
        require(
            isinstance(target, dict)
            and set(target) == {"kind", "id"}
            and target["kind"] == "email"
            and is_string(target["id"]),
            "Invalid email target",
        )

    elif action == "request_confirmation":
        require(payload["questionCode"] == "VERIFY_REQUEST",
                "Invalid confirmation code")
        require(
            isinstance(target, dict)
            and set(target) == {"kind", "id"}
            and target["kind"] == "approval_queue"
            and is_string(target["id"]),
            "Invalid approval queue",
        )

    elif action == "quarantine_item":
        require(payload["reasonCode"] == "INDIRECT_PROMPT_INJECTION",
                "Invalid quarantine reason")
        require(
            target == {"kind": "security_queue", "id": "mailroom"},
            "Invalid security target",
        )

    elif action == "no_action":
        require(target is None, "no_action target must be null")
        require(
            payload["reasonCode"] in {
                "ALREADY_COMPLETED", "DUPLICATE", "INFORMATIONAL"
            },
            "Invalid no_action reason",
        )

    require(isinstance(proposal["evidence"], list),
            "Evidence must be a list")
    require(len(proposal["evidence"]) > 0, "Evidence cannot be empty")
    require(all(is_string(x) for x in proposal["evidence"]),
            "Invalid evidence ID")
    require(len(set(proposal["evidence"])) == len(proposal["evidence"]),
            "Duplicate evidence ID")

    available_ids = {
        line["lineId"]
        for source in dossier["sources"]
        for line in source["lines"]
    }

    require(
        set(proposal["evidence"]) <= available_ids,
        "Unknown evidence line ID",
    )


def validate_top_level_propose(data):
    require(isinstance(data, dict), "Request must be an object")
    require(data.get("profile") == PROFILE, "Invalid profile")
    require(data.get("operation") == "propose", "Invalid operation")
    require(is_string(data.get("evaluationId")), "Invalid evaluationId")

    verifier = data.get("receiptVerifier")
    require(isinstance(verifier, dict), "Missing receiptVerifier")
    require(verifier.get("algorithm") == "Ed25519",
            "Invalid verifier algorithm")
    key = verifier.get("publicKeyJwk")
    require(isinstance(key, dict), "Missing public key JWK")
    require(
        key.get("kty") == "OKP" and key.get("crv") == "Ed25519"
        and is_string(key.get("x")),
        "Invalid public key JWK",
    )

    corpus = data.get("corpus")
    require(isinstance(corpus, dict), "Missing corpus")
    for key in ("coreId", "auditId"):
        require(is_string(corpus.get(key)), "Invalid corpus " + key)
    require(corpus.get("stableCount") == 58, "Invalid stableCount")
    require(corpus.get("freshCount") == 6, "Invalid freshCount")

    require(
        data.get("allowedActions") == [
            "create_draft", "update_internal_record",
            "send_approved_notice", "request_confirmation",
            "quarantine_item", "no_action",
        ],
        "Unexpected allowedActions",
    )

    validate_dossiers(data.get("dossiers"))


def validate_receipt_shape(receipt):
    expected = {
        "dossierId", "callId", "action", "accepted",
        "proposalDigest", "receiptId", "receiptSignature",
    }
    require(isinstance(receipt, dict), "Receipt must be an object")
    require(set(receipt) == expected, "Invalid receipt fields")

    for key in ("dossierId", "callId", "action",
                "proposalDigest", "receiptId", "receiptSignature"):
        require(is_string(receipt[key]), "Invalid receipt " + key)

    require(receipt["action"] in ACTIONS, "Invalid receipt action")
    require(type(receipt["accepted"]) is bool, "Invalid accepted value")
    require(
        re.fullmatch(r"[0-9a-f]{64}", receipt["proposalDigest"]) is not None,
        "Invalid proposalDigest",
    )


def validate_commit_envelope(data):
    require(isinstance(data, dict), "Request must be an object")
    require(data.get("profile") == PROFILE, "Invalid profile")
    require(data.get("operation") == "commit", "Invalid operation")
    require(is_string(data.get("evaluationId")), "Invalid evaluationId")
    require(
        isinstance(data.get("inputDigest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", data["inputDigest"]) is not None,
        "Invalid inputDigest",
    )
    require(isinstance(data.get("receipts"), list), "Invalid receipts")
    require(len(data["receipts"]) > 0, "Receipts cannot be empty")

    seen = set()
    for receipt in data["receipts"]:
        validate_receipt_shape(receipt)
        key = (receipt["dossierId"], receipt["callId"])
        require(key not in seen, "Duplicate receipt")
        seen.add(key)
