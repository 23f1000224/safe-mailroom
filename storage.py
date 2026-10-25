
import json
import os
from contextlib import contextmanager

from sqlalchemy import (
    Column, Integer, String, Text, create_engine, select
)
from sqlalchemy.orm import declarative_base, sessionmaker

from security import canonical_json

Base = declarative_base()


def normalize_database_url(url):
    if url.startswith("postgres://"):
        return url.replace(
            "postgres://", "postgresql+psycopg://", 1
        )
    if url.startswith("postgresql://"):
        return url.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    return url


database_url = normalize_database_url(
    os.getenv("DATABASE_URL", "sqlite:///mailroom.db")
)

engine_options = {"pool_pre_ping": True}

if database_url.startswith("sqlite:"):
    engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(database_url, **engine_options)
SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


class Evaluation(Base):
    __tablename__ = "evaluations"

    evaluation_id = Column(String(256), primary_key=True)
    request_hash = Column(String(64), nullable=False)
    input_digest = Column(String(64), nullable=False)
    verifier_jwk = Column(Text, nullable=False)
    propose_response = Column(Text, nullable=False)
    commit_request_hash = Column(String(64), nullable=True)
    commit_response = Column(Text, nullable=True)


class Proposal(Base):
    __tablename__ = "proposals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(String(256), nullable=False, index=True)
    dossier_id = Column(String(256), nullable=False)
    fingerprint = Column(String(64), nullable=False)
    call_id = Column(String(128), nullable=False)
    proposal_json = Column(Text, nullable=False)
    proposal_digest = Column(String(64), nullable=False)

    __table_args__ = (
        # SQLite/PostgreSQL both support a unique constraint through
        # a database index created below.
    )


class DecisionCache(Base):
    __tablename__ = "decision_cache"

    fingerprint = Column(String(64), primary_key=True)
    proposal_json = Column(Text, nullable=False)


class SavedReceipt(Base):
    __tablename__ = "saved_receipts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(String(256), nullable=False, index=True)
    dossier_id = Column(String(256), nullable=False)
    call_id = Column(String(128), nullable=False)
    receipt_id = Column(String(256), nullable=False)
    receipt_json = Column(Text, nullable=False)
    outcome = Column(String(16), nullable=False)


# A database uniqueness constraint protects receipt replay and duplicates.
from sqlalchemy import UniqueConstraint

SavedReceipt.__table__.append_constraint(
    UniqueConstraint(
        "evaluation_id", "dossier_id", name="uq_receipt_dossier"
    )
)

Proposal.__table__.append_constraint(
    UniqueConstraint(
        "evaluation_id", "dossier_id", name="uq_proposal_dossier"
    )
)


def init_db():
    Base.metadata.create_all(bind=engine)


@contextmanager
def transaction():
    session = SessionLocal()
    try:
        with session.begin():
            yield session
    finally:
        session.close()


def json_load(value):
    return json.loads(value)


def get_evaluation(session, evaluation_id):
    return session.get(Evaluation, evaluation_id)


def get_proposal(session, evaluation_id, dossier_id):
    return session.execute(
        select(Proposal).where(
            Proposal.evaluation_id == evaluation_id,
            Proposal.dossier_id == dossier_id,
        )
    ).scalar_one_or_none()


def get_cached_proposal(session, fingerprint):
    item = session.get(DecisionCache, fingerprint)
    return json_load(item.proposal_json) if item else None


def save_cached_proposal(session, fingerprint, proposal):
    existing = session.get(DecisionCache, fingerprint)

    if existing is None:
        session.add(
            DecisionCache(
                fingerprint=fingerprint,
                proposal_json=canonical_json(proposal),
            )
        )


def save_evaluation(session, evaluation_id, request_hash,
                    input_digest, verifier_jwk, response,
                    proposals):
    evaluation = Evaluation(
        evaluation_id=evaluation_id,
        request_hash=request_hash,
        input_digest=input_digest,
        verifier_jwk=canonical_json(verifier_jwk),
        propose_response=canonical_json(response),
    )
    session.add(evaluation)

    for item in proposals:
        session.add(
            Proposal(
                evaluation_id=evaluation_id,
                dossier_id=item["proposal"]["dossierId"],
                fingerprint=item["fingerprint"],
                call_id=item["proposal"]["callId"],
                proposal_json=canonical_json(item["proposal"]),
                proposal_digest=item["proposalDigest"],
            )
        )


def list_proposals(session, evaluation_id):
    rows = session.execute(
        select(Proposal)
        .where(Proposal.evaluation_id == evaluation_id)
        .order_by(Proposal.id)
    ).scalars().all()

    return [
        {
            "dossierId": row.dossier_id,
            "callId": row.call_id,
            "proposal": json_load(row.proposal_json),
            "proposalDigest": row.proposal_digest,
        }
        for row in rows
    ]


def save_receipts_and_commit_response(
    session, evaluation, receipts, outcomes, request_hash
):
    for receipt, outcome in zip(receipts, outcomes):
        session.add(
            SavedReceipt(
                evaluation_id=evaluation.evaluation_id,
                dossier_id=receipt["dossierId"],
                call_id=receipt["callId"],
                receipt_id=receipt["receiptId"],
                receipt_json=canonical_json(receipt),
                outcome=outcome["status"],
            )
        )

    response = {
        "profile": "ga5-mailroom-action-gate/v2",
        "evaluationId": evaluation.evaluation_id,
        "status": "completed",
        "inputDigest": evaluation.input_digest,
        "outcomes": outcomes,
    }

    evaluation.commit_request_hash = request_hash
    evaluation.commit_response = canonical_json(response)

    return response
