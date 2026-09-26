"""Apply the Orion adapter only to the audited upstream FCC source revision."""

import hashlib
import sysconfig
from pathlib import Path

EXPECTED = "09955d7712b618df8b368c21d9bba4320fa0af70e7250e1ca5f2e79bddfccaef"


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


if __name__ == "__main__":
    patch_routes(Path(sysconfig.get_paths()["purelib"]) / "api/routes.py")
