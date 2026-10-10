import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import jwt
import pytest

from common.token_decorator import check_token
from services.user_service import decode_jwt_token, generate_jwt_token


def _make_request(token=None):
    headers = {}
    if token is not None:
        headers["Authorization"] = token
    return SimpleNamespace(headers=headers, ctx=SimpleNamespace())


def _run_check_token(request):
    @check_token
    async def handler(request):
        return SimpleNamespace(status=200, body=b"ok")

    return asyncio.run(handler(request))


def test_generate_jwt_token_raises_without_secret(monkeypatch):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError):
        asyncio.run(generate_jwt_token("1", "alice"))


def test_generate_jwt_token_raises_with_empty_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "")

    with pytest.raises(RuntimeError):
        asyncio.run(generate_jwt_token("1", "alice"))


def test_generate_and_decode_round_trip_succeeds(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")

    token = asyncio.run(generate_jwt_token("1", "alice", role="admin"))
    payload = asyncio.run(decode_jwt_token(token))

    assert payload["id"] == "1"
    assert payload["username"] == "alice"
    assert payload["role"] == "admin"


def test_decode_jwt_token_missing_secret_returns_500(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    token = asyncio.run(generate_jwt_token("1", "alice"))

    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    result = asyncio.run(decode_jwt_token(token))

    assert result == (None, 500, "JWT_SECRET_KEY environment variable is not set")


def test_decode_jwt_token_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "right-secret")
    token = asyncio.run(generate_jwt_token("1", "alice"))

    monkeypatch.setenv("JWT_SECRET_KEY", "wrong-secret")
    result = asyncio.run(decode_jwt_token(token))

    assert result[0] is None
    assert result[1] == 400


def test_decode_jwt_token_rejects_expired_token(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    payload = {
        "id": "1",
        "username": "alice",
        "role": "user",
        "exp": datetime.utcnow() - timedelta(days=1),
    }
    token = jwt.encode(payload, "test-secret", algorithm="HS256")

    result = asyncio.run(decode_jwt_token(token))

    assert result[0] is None
    assert result[1] == 401


def test_decode_jwt_token_rejects_malformed_token(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")

    result = asyncio.run(decode_jwt_token("not-a-jwt"))

    assert result[0] is None
    assert result[1] == 400


def test_check_token_rejects_missing_authorization_header(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    request = _make_request(token=None)

    response = _run_check_token(request)

    assert response.status == 401


def test_check_token_returns_500_when_secret_missing(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    token = asyncio.run(generate_jwt_token("1", "alice"))

    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    request = _make_request(token=f"Bearer {token}")

    response = _run_check_token(request)

    assert response.status == 500


def test_check_token_allows_valid_token_through(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    token = asyncio.run(generate_jwt_token("1", "alice"))
    request = _make_request(token=f"Bearer {token}")

    response = _run_check_token(request)

    assert response.status == 200
    assert request.ctx.user_payload["id"] == "1"


def test_check_token_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    request = _make_request(token="Bearer not-a-jwt")

    response = _run_check_token(request)

    assert response.status == 401


def test_check_token_rejects_expired_token(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    payload = {
        "id": "1",
        "username": "alice",
        "role": "user",
        "exp": datetime.utcnow() - timedelta(days=1),
    }
    token = jwt.encode(payload, "test-secret", algorithm="HS256")
    request = _make_request(token=f"Bearer {token}")

    response = _run_check_token(request)

    assert response.status == 401
