"""Apply the Orion adapter only to the audited upstream FCC source revision."""

import hashlib
import sysconfig
from pathlib import Path

EXPECTED = "09955d7712b618df8b368c21d9bba4320fa0af70e7250e1ca5f2e79bddfccaef"
LEDGER_EXPECTED = "a6a2c1796c7165de4ea5edbfcc1a75423b546ace32343208b6077ac8296edd5f"


def patch_routes(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED:
        raise RuntimeError(
            "FCC routes changed; review messages transport patch before updating the pin"
        )
    text = source.decode()
    text = text.replace(
        "async def create_message(\n    request_data: MessagesRequest,",
        "async def create_message(\n    request_data: MessagesRequest,\n    request: Request,",
    )
    text = text.replace(
        "from . import dependencies",
        "from orion_fcc_messages_transport import adapt_message_response\n\nfrom . import dependencies",
    )
    text = text.replace(
        '"""Create a message (always streaming)."""\n    return handler.create(request_data)',
        '"""Honor the Messages wire format and maintain SSE liveness."""\n'
        "    return await adapt_message_response(\n"
        "        handler.create(request_data),\n"
        '        stream="stream" in request_data.model_fields_set and request_data.stream is True,\n'
        "        disconnected=request.is_disconnected,\n"
        "    )",
    )
    path.write_text(text)


def patch_error_emitter(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != LEDGER_EXPECTED:
        raise RuntimeError(
            "FCC ledger changed; review error transport patch before updating the pin"
        )
    text = source.decode().replace(
        "        error_index = self.blocks.allocate_index()\n"
        '        yield self.content_block_start(error_index, "text")\n'
        '        yield self.content_block_delta(error_index, "text_delta", error_message)\n'
        "        yield self.content_block_stop(error_index)",
        "        yield self.emit_top_level_error(error_message)",
    )
    path.write_text(text)


if __name__ == "__main__":
    root = Path(sysconfig.get_paths()["purelib"])
    patch_routes(root / "api/routes.py")
    patch_error_emitter(root / "core/anthropic/streaming/ledger.py")
