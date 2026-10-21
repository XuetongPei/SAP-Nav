"""Shared model requests must serialize and survive client process termination."""
import multiprocessing as mp
import time
from types import SimpleNamespace

from vlfm.vlm import server_wrapper


def test_model_clients_are_serialized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = mp.get_context("fork")
    active, maximum = ctx.Value("i", 0), ctx.Value("i", 0)
    guard = ctx.Lock()

    def post(*args, **kwargs):
        with guard:
            active.value += 1
            maximum.value = max(maximum.value, active.value)
        time.sleep(0.05)
        with guard:
            active.value -= 1
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"ok": True})

    monkeypatch.setattr(server_wrapper.requests, "post", post)
    clients = [ctx.Process(target=server_wrapper._send_request, args=("http://localhost:65530/test",)) for _ in range(4)]
    try:
        for client in clients:
            client.start()
        for client in clients:
            client.join(5)
            assert client.exitcode == 0
        assert maximum.value == 1
    finally:
        for client in clients:
            if client.is_alive():
                client.terminate()
                client.join()


def test_terminated_client_does_not_leave_stale_lock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = mp.get_context("fork")
    entered = ctx.Event()

    def blocked_post(*args, **kwargs):
        entered.set()
        time.sleep(60)

    monkeypatch.setattr(server_wrapper.requests, "post", blocked_post)
    url = "http://localhost:65530/test"
    first = ctx.Process(target=server_wrapper._send_request, args=(url,))
    first.start()
    try:
        assert entered.wait(5)
    finally:
        first.terminate()
        first.join()
    monkeypatch.setattr(server_wrapper.requests, "post", lambda *a, **k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"ok": True}))
    second = ctx.Process(target=server_wrapper._send_request, args=(url,))
    second.start()
    try:
        second.join(5)
        assert second.exitcode == 0
    finally:
        if second.is_alive():
            second.terminate()
            second.join()
