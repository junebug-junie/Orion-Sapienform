import subprocess
from pathlib import Path

from .settings import Settings

def render_modules_conf() -> str:
    """
    Minimal modules.conf so Asterisk doesn't die on startup.
    Let it autoload and resolve deps.
    """
    return """[modules]
autoload=yes
"""


def render_rtp_conf() -> str:
    """
    Minimal rtp.conf so RTP has sane defaults.
    """
    return """[general]
rtpstart=10000
rtpend=20000
icesupport=yes
"""


def write_if_missing(path: Path, content: str) -> None:
    """Write content only if file doesn't exist or is empty."""
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_pjsip_conf(settings: Settings) -> str:
    """Generate a minimal pjsip.conf for one endpoint."""
    codecs_line = ",".join(settings.codecs_list) or "ulaw"
    ext = settings.sip_ext
    return f"""[transport-udp]
type=transport
protocol=udp
bind=0.0.0.0

[{ext}]
type=endpoint
transport=transport-udp
context=local
disallow=all
allow={codecs_line}
auth={ext}
aors={ext}

[{ext}]
type=auth
auth_type=userpass
username={ext}
password={settings.sip_secret}

[{ext}]
type=aor
max_contacts=1
"""


def render_extensions_conf(settings: Settings) -> str:
    """Generate extensions.conf with echo test and optional page/auto-answer."""
    lines = [
        "[local]",
        "exten => 600,1,Answer()",
        " same => n,Echo()",
        " same => n,Hangup()",
        "",
    ]
    if settings.autoanswer:
        ext = settings.sip_ext
        lines += [
            f"exten => 700,1,NoOp(Page {ext} with auto-answer)",
            " same => n,Set(PJSIP_HEADER(add,Call-Info)=<sip:autoanswer>)",
            f" same => n,Dial(PJSIP/{ext},20)",
            " same => n,Hangup()",
        ]
    return "\n".join(lines) + "\n"


def render_sep_xml(settings: Settings) -> str:
    """Cisco SEP XML for the 8831, in SIP mode pointing at LAN host IP."""
    host_ip = str(settings.lan_host_ip)
    ext = settings.sip_ext
    secret = settings.sip_secret
    autoanswer_enabled = 2 if settings.autoanswer else 0

    return f"""<device>
  <deviceProtocol>SIP</deviceProtocol>
  <devicePool>
    <callManagerGroup>
      <members>
        <member priority="0">
          <callManager>
            <processNodeName>{host_ip}</processNodeName>
            <ports>
              <sipPort>5060</sipPort>
            </ports>
          </callManager>
        </member>
      </members>
    </callManagerGroup>
  </devicePool>
  <sipProfile>
    <sipProxies>
      <backupProxy></backupProxy>
      <proxy>{host_ip}</proxy>
      <emergencyProxy></emergencyProxy>
    </sipProxies>
    <sipLines>
      <line button="1">
        <featureID>9</featureID>
        <name>{ext}</name>
        <displayName>Orion Conf</displayName>
        <authName>{ext}</authName>
        <authPassword>{secret}</authPassword>
        <contact>{ext}</contact>
        <proxy>{host_ip}</proxy>
        <port>5060</port>
      </line>
    </sipLines>
    <autoAnswerTimer>1</autoAnswerTimer>
    <autoAnswerEnabled>{autoanswer_enabled}</autoAnswerEnabled>
  </sipProfile>
</device>
"""


def bootstrap_asterisk_and_cisco(settings: Settings) -> None:
    """
    Ensure Asterisk dirs exist, write core configs and SEP<MAC>.cnf.xml (only if missing).
    """
    # Ensure dirs
    for p in [
        settings.asterisk_etc_dir,
        Path("/var/lib/asterisk"),
        Path("/var/log/asterisk"),
        Path("/var/spool/asterisk"),
        settings.tftp_root,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    # Core Asterisk configs
    modules_path = settings.asterisk_etc_dir / "modules.conf"
    rtp_path = settings.asterisk_etc_dir / "rtp.conf"
    pjsip_path = settings.asterisk_etc_dir / "pjsip.conf"
    ext_path = settings.asterisk_etc_dir / "extensions.conf"

    write_if_missing(modules_path, render_modules_conf())
    write_if_missing(rtp_path, render_rtp_conf())
    write_if_missing(pjsip_path, render_pjsip_conf(settings))
    write_if_missing(ext_path, render_extensions_conf(settings))

    # Cisco SEP config
    sep_file = settings.tftp_root / settings.cisco_sep_filename
    if not sep_file.exists() or sep_file.stat().st_size == 0:
        sep_file.write_text(render_sep_xml(settings), encoding="utf-8")
        print(f"[VOIP] Wrote Cisco SEP config: {sep_file}", flush=True)
    else:
        print(f"[VOIP] Using existing Cisco SEP config: {sep_file}", flush=True)



def start_tftp(root: Path) -> subprocess.Popen:
    """Start in.tftpd (tftpd-hpa) serving /tftpboot."""
    root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/sbin/in.tftpd",
        "--listen",
        "--address",
        "0.0.0.0:69",
        "--secure",
        str(root),
    ]
    print(f"[VOIP] Starting TFTP: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def start_asterisk() -> subprocess.Popen:
    """Start Asterisk in the foreground (we keep the Popen object)."""
    cmd = ["/usr/sbin/asterisk", "-f", "-vvv"]
    print(f"[VOIP] Starting Asterisk: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def asterisk_cmd(command: str) -> subprocess.CompletedProcess:
    """Run an Asterisk CLI command and capture output."""
    full = ["/usr/sbin/asterisk", "-rx", command]
    print(f"[VOIP] asterisk -rx {command}", flush=True)
    return subprocess.run(full, capture_output=True, text=True)
