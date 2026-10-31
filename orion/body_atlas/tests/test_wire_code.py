"""HTTP calls and Redis keys named in code (wire_code.py): each rule, and the false shapes an
earlier draft claimed."""
from __future__ import annotations

from pathlib import Path

from orion.body_atlas import wire_code as W


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _calls(refs: W.WireRefs) -> set[tuple[str, str]]:
    return {(r.service, r.value) for r in refs.http}


def _keys(refs: W.WireRefs) -> set[tuple[str, str, str]]:
    return {(r.service, k, r.relation) for k, rs in refs.redis.items() for r in rs}


def test_url_host_and_compose_prefix():
    assert W.url_host("http://orion-athena-durable-runs:8121/physiology/bid") == "orion-athena-durable-runs"
    assert W.url_host("http://${PROJECT}-vector-host:8320/embedding") == "vector-host"
    assert W.url_host("redis://100.92.216.81:6379/0") is None  # not HTTP: the bus, not a call
    assert W.url_host("see http://x in the docs") is None  # a sentence, not a URL


def test_a_literal_url_in_service_code_is_a_call_and_a_docstring_is_not(tmp_path):
    _w(tmp_path, "services/orion-a/app/main.py",
       '"""Talks to http://orion-athena-c:9/ in prose."""\nURL = "http://orion-athena-b:8121/x"\n')
    assert _calls(W.code_wire_refs(tmp_path)) == {("orion-a", "http://orion-athena-b:8121/x")}


def test_a_settings_url_counts_only_where_the_field_is_read_and_env_wins(tmp_path):
    _w(tmp_path, "services/orion-a/app/settings.py",
       "from pydantic_settings import BaseSettings\nfrom pydantic import Field\n"
       "class Settings(BaseSettings):\n"
       '    notify_url: str = Field("http://orion-notify:7140", alias="NOTIFY_URL")\n'
       '    unused_url: str = Field("http://orion-unused:1", alias="UNUSED_URL")\n'
       "settings = Settings()\n")
    _w(tmp_path, "services/orion-a/app/main.py",
       "from app.settings import settings\nimport httpx\ndef go():\n    return httpx.post(settings.notify_url)\n")
    _w(tmp_path, "services/orion-a/.env_example", "NOTIFY_URL=http://orion-athena-notify:7140\n")
    calls = _calls(W.code_wire_refs(tmp_path))
    assert ("orion-a", "http://orion-athena-notify:7140") in calls  # the env value, read by main.py
    assert ("orion-a", "http://orion-unused:1") not in calls  # a field nobody reads is config, not a call


def test_getattr_reads_a_settings_field(tmp_path):
    _w(tmp_path, "services/orion-a/app/settings.py",
       "from pydantic_settings import BaseSettings\n"
       "class Settings(BaseSettings):\n    mind_base_url: str = 'http://mind:6611'\n")
    _w(tmp_path, "services/orion-a/app/x.py", "def f(s):\n    return getattr(s, 'mind_base_url', '')\n")
    assert ("orion-a", "http://mind:6611") in _calls(W.code_wire_refs(tmp_path))


def test_a_shared_client_reaches_only_the_services_that_use_it(tmp_path):
    _w(tmp_path, "orion/__init__.py", "")
    _w(tmp_path, "orion/phys/__init__.py", "")
    _w(tmp_path, "orion/phys/client.py",
       'DEFAULT_URL = "http://orion-athena-durable-runs:8121/physiology/bid"\n'
       "def bid(w):\n    return post(DEFAULT_URL, w)\n"
       "def unrelated():\n    return 1\n")
    _w(tmp_path, "orion/phys/wrap.py", "from orion.phys.client import bid\ndef want(x):\n    return bid(x)\n")
    _w(tmp_path, "services/orion-a/app/main.py", "from orion.phys.client import bid\nbid(1)\n")
    _w(tmp_path, "services/orion-b/app/main.py", "from orion.phys.client import unrelated\n")
    _w(tmp_path, "services/orion-c/app/main.py", "from orion.phys.wrap import want\n")  # two hops
    calls = _calls(W.code_wire_refs(tmp_path))
    url = "http://orion-athena-durable-runs:8121/physiology/bid"
    assert ("orion-a", url) in calls and ("orion-c", url) in calls
    assert not any(s == "orion-b" for s, _ in calls)


def test_a_model_field_url_is_a_call_only_for_code_that_reads_the_field(tmp_path):
    """Review finding (live 2026-10-30): orion-policy-runtime imports a validator that names
    VisualBaselinePolicy, whose thought_url default is a URL; it never calls orion-thought.
    orion-proposal-runtime reads policy.thought_url and does."""
    _w(tmp_path, "orion/__init__.py", "")
    _w(tmp_path, "orion/rev.py",
       "from pydantic import BaseModel\n"
       "class Policy(BaseModel):\n    thought_url: str = 'http://orion-athena-thought:7155'\n"
       "def validate(p: Policy):\n    return p.enabled\n"
       "def load():\n    return Policy()\n")
    _w(tmp_path, "services/orion-policy/app/main.py", "from orion.rev import validate\n")
    _w(tmp_path, "services/orion-proposal/app/main.py",
       "from orion.rev import load\nurlopen(load().thought_url + '/x')\n")
    calls = _calls(W.code_wire_refs(tmp_path))
    assert ("orion-proposal", "http://orion-athena-thought:7155") in calls
    assert not any(s == "orion-policy" for s, _ in calls)


def test_env_key_read_in_shared_code_resolves_per_importing_service(tmp_path):
    _w(tmp_path, "orion/__init__.py", "")
    _w(tmp_path, "orion/c.py", "import os\ndef base():\n    return os.getenv('PEER_URL')\n")
    _w(tmp_path, "services/orion-a/app/main.py", "from orion.c import base\n")
    _w(tmp_path, "services/orion-a/.env_example", "PEER_URL=http://orion-athena-peer:1\n")
    _w(tmp_path, "services/orion-b/app/main.py", "from orion.c import base\n")  # its env names nothing
    assert _calls(W.code_wire_refs(tmp_path)) == {("orion-a", "http://orion-athena-peer:1")}


# --- Redis -------------------------------------------------------------------------------------

def test_redis_keys_by_verb_through_constants_and_handles(tmp_path):
    _w(tmp_path, "orion/__init__.py", "")
    _w(tmp_path, "orion/schemas/__init__.py", "")
    _w(tmp_path, "orion/schemas/reg.py", 'REG_KEY = "orion:regulation:latest"\n')
    _w(tmp_path, "orion/read.py",
       "import redis\nfrom orion.schemas.reg import REG_KEY\n"
       "def read_it(url):\n    client = redis.Redis.from_url(url)\n    return client.get(REG_KEY)\n")
    _w(tmp_path, "services/orion-w/app/main.py",
       "from orion.schemas.reg import REG_KEY\nasync def tick(rpc_bus):\n"
       "    await rpc_bus.redis.setex(REG_KEY, 300, 'x')\n")
    _w(tmp_path, "services/orion-r/app/main.py", "from orion.read import read_it\n")
    assert _keys(W.code_wire_refs(tmp_path)) == {
        ("orion-w", "orion:regulation:latest", "writes"), ("orion-r", "orion:regulation:latest", "reads")}


def test_a_reply_or_a_dict_is_not_a_redis_handle(tmp_path):
    """Review findings (2026-10-30 draft): ``persistence = await redis.info(...)`` and
    ``msg = await pubsub.get_message()`` made every ``.get("...")`` on the reply a "key"
    (``rdb_current_bgsave_time_sec``, ``data``, ``channel`` read by 80 services)."""
    _w(tmp_path, "services/orion-g/app/main.py",
       "async def check(redis):\n"
       "    persistence = await redis.info('persistence')\n"
       "    a = persistence.get('rdb_current_bgsave_time_sec')\n"
       "    pubsub = redis.pubsub()\n"
       "    msg = await pubsub.get_message()\n"
       "    b = msg.get('data')\n"
       "    row = {'node': 1}\n"
       "    c = row.get('node')\n"
       "    await redis.xadd('orion:stream:x', {'a': 1})\n"  # a stream: the bus, not a key
       "    await redis.publish('orion:chan', 'm')\n"
       "    return a, b, c\n")
    refs = W.code_wire_refs(tmp_path)
    assert refs.redis == {}


def test_pipeline_and_annotated_handles(tmp_path):
    _w(tmp_path, "services/orion-v/app/main.py",
       "from redis.asyncio import Redis\nKEY = 'orion:vision:look:doorbell'\n"
       "async def f(r: Redis):\n    pipe = r.pipeline()\n    pipe.exists(KEY)\n"
       "async def g(client: 'Redis'):\n    await client.set(KEY, 1)\n")
    assert _keys(W.code_wire_refs(tmp_path)) == {
        ("orion-v", "orion:vision:look:doorbell", "reads"), ("orion-v", "orion:vision:look:doorbell", "writes")}


def test_key_families_only_from_a_named_prefix_and_the_rest_is_dynamic(tmp_path):
    _w(tmp_path, "services/orion-h/app/main.py",
       "PREFIX = 'orion:retro:presence:browser:'\n"
       "async def f(redis, sid, x):\n"
       "    await redis.set(f'{PREFIX}{sid}', 1)\n"  # a family: one named prefix
       "    await redis.get(f'orion:literal:{sid}')\n"  # a literal prefix: dynamic
       "    await redis.get(x)\n")  # a parameter: dynamic
    refs = W.code_wire_refs(tmp_path)
    assert _keys(refs) == {("orion-h", "orion:retro:presence:browser:*", "writes")}
    assert refs.dynamic == {"orion-h": ["services/orion-h/app/main.py"]}


def test_a_settings_redis_key_resolves_through_env(tmp_path):
    _w(tmp_path, "services/orion-s/app/settings.py",
       "from pydantic_settings import BaseSettings\n"
       "class Settings(BaseSettings):\n    state_key: str = 'orion:default'\n")
    _w(tmp_path, "services/orion-s/app/main.py",
       "async def f(redis, settings):\n    await redis.hgetall(settings.state_key)\n")
    _w(tmp_path, "services/orion-s/.env_example", "STATE_KEY=orion:state:live\n")
    assert _keys(W.code_wire_refs(tmp_path)) == {("orion-s", "orion:state:live", "reads")}


def test_tests_evals_and_the_atlas_itself_are_skipped(tmp_path):
    _w(tmp_path, "services/orion-a/tests/test_x.py", "U = 'http://orion-athena-b:1'\n")
    _w(tmp_path, "services/orion-a/evals/run.py", "U = 'http://orion-athena-b:1'\n")
    _w(tmp_path, "orion/body_atlas/x.py", "U = 'http://orion-athena-b:1'\n")
    assert W.code_wire_refs(tmp_path).http == []


def test_a_parenthesized_import_with_comments_still_reaches_the_shared_symbol(tmp_path):
    """The no-parse import reader (a module that names no URL itself): a comment inside
    ``from m import (\\n a,  # why\\n b,\\n)`` must not hide ``b``."""
    _w(tmp_path, "orion/__init__.py", "")
    _w(tmp_path, "orion/c.py", "U = 'http://orion-athena-peer:1'\ndef a():\n    return 1\ndef b():\n    return U\n")
    _w(tmp_path, "services/orion-a/app/main.py", "from orion.c import (\n    a,  # first\n    b,\n)\n")
    assert _calls(W.code_wire_refs(tmp_path)) == {("orion-a", "http://orion-athena-peer:1")}


# --- review findings (2026-10-31) --------------------------------------------------------------

def test_a_settings_default_the_env_overrides_is_not_counted(tmp_path):
    _w(tmp_path, "services/orion-a/app/settings.py",
       "from pydantic_settings import BaseSettings\nfrom pydantic import Field\n"
       "class Settings(BaseSettings):\n"
       '    notify_url: str = Field("http://orion-notify:7140", alias="NOTIFY_URL")\n')
    _w(tmp_path, "services/orion-a/app/main.py", "def go(settings):\n    return settings.notify_url\n")
    _w(tmp_path, "services/orion-a/.env_example", "NOTIFY_URL=http://orion-athena-recall:8090\n")
    assert _calls(W.code_wire_refs(tmp_path)) == {("orion-a", "http://orion-athena-recall:8090")}


def test_augmented_parameter_and_loop_names_are_not_one_constant(tmp_path):
    _w(tmp_path, "services/orion-a/app/main.py",
       "import redis\nr = redis.Redis()\n"
       "def f():\n    key = 'orion:base'\n    key += ':suffix'\n    r.set(key, 1)\n"
       "def g(name):\n    r.get(name)\n"
       "def h():\n    name = 'orion:x'\n    return name\n")
    refs = W.code_wire_refs(tmp_path)
    assert refs.redis == {} and refs.dynamic == {"orion-a": ["services/orion-a/app/main.py"]}


def test_a_name_bound_to_redis_and_to_an_http_client_is_not_a_handle(tmp_path):
    _w(tmp_path, "services/orion-a/app/main.py",
       "import redis, httpx\n"
       "def f():\n    client = redis.Redis()\n    return client\n"
       "def g():\n    client = httpx.Client()\n    client.get('/health')\n    client.delete('orion:thing')\n")
    assert W.code_wire_refs(tmp_path).redis == {}


def test_a_request_path_on_a_redis_named_object_is_not_a_key(tmp_path):
    _w(tmp_path, "services/orion-a/app/main.py", "def f(redis):\n    redis.get('/health')\n")
    assert W.code_wire_refs(tmp_path).redis == {}


def test_a_name_re_exported_by_a_package_init_still_reaches_its_act(tmp_path):
    _w(tmp_path, "orion/__init__.py", "")
    _w(tmp_path, "orion/pkg/__init__.py", "from .impl import write\n")
    _w(tmp_path, "orion/pkg/impl.py", "import redis\ndef write():\n    redis.Redis().set('orion:k', 1)\n")
    _w(tmp_path, "services/orion-a/app/main.py", "from orion.pkg import write\n")
    assert _keys(W.code_wire_refs(tmp_path)) == {("orion-a", "orion:k", "writes")}


def test_a_shared_prefix_with_a_literal_suffix_is_the_exact_key(tmp_path):
    _w(tmp_path, "services/orion-a/app/main.py",
       "PREFIX = 'orion:fam'\nasync def f(redis):\n    await redis.set(f'{PREFIX}:latest', 1)\n"
       "    await redis.set(PREFIX + ':other', 1)\n")
    assert _keys(W.code_wire_refs(tmp_path)) == {("orion-a", "orion:fam:latest", "writes"),
                                                 ("orion-a", "orion:fam:other", "writes")}


def test_os_environ_subscript_is_an_env_read(tmp_path):
    _w(tmp_path, "services/orion-a/app/main.py", "import os\nURL = os.environ['PEER_URL']\n")
    _w(tmp_path, "services/orion-a/.env_example", "PEER_URL=http://orion-athena-peer:1\n")
    assert _calls(W.code_wire_refs(tmp_path)) == {("orion-a", "http://orion-athena-peer:1")}
