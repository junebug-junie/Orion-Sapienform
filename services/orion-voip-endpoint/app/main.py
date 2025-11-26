import os
import signal
import sys
import subprocess

import uvicorn

from .settings import Settings
from .asterisk_control import (
    bootstrap_asterisk_and_cisco,
    start_tftp,
    start_asterisk,
    asterisk_cmd,
)
from .bus_integration import (
    init_bus,
    make_bus_publish,
    start_bus_listener_thread,
)
from .http_api import create_app


settings = Settings()

# Child processes (TFTP + Asterisk)
_tftp_proc: subprocess.Popen | None = None
_ast_proc: subprocess.Popen | None = None


def build_action_handlers(bus_publish):
    """
    Define the core actions that both HTTP and bus can invoke.
    """

    def do_echo_call() -> dict:
        cmd = f"channel originate PJSIP/{settings.sip_ext} extension 600@local"
        result = asterisk_cmd(cmd)
        ok = result.returncode == 0
        bus_publish(
            "echo_call",
            ok=ok,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
        )
        if not ok:
            raise RuntimeError(result.stderr or "asterisk echo_call failed")
        return {"status": "ok", "command": cmd}

    def do_page_echo() -> dict:
        cmd = "channel originate Local/700@local extension 600@local"
        result = asterisk_cmd(cmd)
        ok = result.returncode == 0
        bus_publish(
            "page_echo",
            ok=ok,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
        )
        if not ok:
            raise RuntimeError(result.stderr or "asterisk page_echo failed")
        return {"status": "ok", "command": cmd}

    return {
        "echo_call": do_echo_call,
        "page_echo": do_page_echo,
    }, do_echo_call, do_page_echo


def shutdown():
    """Terminate child processes cleanly."""
    global _ast_proc, _tftp_proc
    print("[VOIP] Shutting down...", flush=True)
    for proc, name in [(_ast_proc, "Asterisk"), (_tftp_proc, "TFTP")]:
        if proc and proc.poll() is None:
            print(f"[VOIP] Terminating {name}...", flush=True)
            try:
                proc.terminate()
            except Exception:
                pass


def handle_sig(sig, frame):
    print(f"[VOIP] Received signal {sig}, shutting down...", flush=True)
    shutdown()
    sys.exit(0)


def run():
    global _ast_proc, _tftp_proc

    print("[VOIP] Settings:", settings.summary(), flush=True)

    # 1) Config files
    bootstrap_asterisk_and_cisco(settings)

    # 2) Bus
    bus = init_bus(settings)
    bus_publish = make_bus_publish(bus, settings)
    bus_publish("startup")

    # 3) Start TFTP + Asterisk
    _tftp_proc = start_tftp(settings.tftp_root)
    _ast_proc = start_asterisk()

    # 4) Actions + bus listener
    action_handlers, do_echo_call, do_page_echo = build_action_handlers(bus_publish)
    if bus and bus.enabled:
        start_bus_listener_thread(bus, settings, bus_publish, action_handlers)
    else:
        print("[VOIP] Bus disabled; listener not started", flush=True)

    # 5) HTTP app
    app = create_app(settings, bus_publish, do_echo_call, do_page_echo)

    # 6) Signals
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=settings.api_port,
            log_level="info",
        )
    finally:
        shutdown()


if __name__ == "__main__":
    run()
