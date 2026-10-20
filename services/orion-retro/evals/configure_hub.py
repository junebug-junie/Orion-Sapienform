"""Set the exact operator-selected browser origin and existing private Retro paths."""
import argparse
from pathlib import Path
from urllib.parse import urlsplit


def configure(env, root, origin):
    u=urlsplit(origin)
    if u.scheme not in ('http','https') or not u.hostname or u.username or u.password or u.query or u.fragment or u.path not in ('','/'):
        raise ValueError('origin must be an exact http(s) origin, without path or credentials')
    origin=origin.rstrip('/')
    for path in (root/'hub-auth/hub.token',):
        if not path.is_file():raise ValueError('run install_catalog.py first')
    values={'HUB_RETRO_ENABLED':'true','HUB_RETRO_ORIGIN':origin,
        'HUB_RETRO_SOCKET':'/run/orion-retro/retro.sock','HUB_RETRO_AUTH_DIR':'/run/orion-retro-auth',
        'HUB_RETRO_SOCKET_HOST':str(root/'ipc'),'HUB_RETRO_AUTH_HOST':str(root/'hub-auth')}
    lines=env.read_text().splitlines() if env.exists() else []
    found=set();out=[]
    for line in lines:
        key,sep,_=line.partition('=')
        if sep and key in values:line=key+'='+values[key];found.add(key)
        out.append(line)
    out.extend(k+'='+v for k,v in values.items() if k not in found)
    env.write_text('\n'.join(out)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--env',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path.home()/'.local/share/orion/retro/service')
    p.add_argument('--origin',required=True);a=p.parse_args();configure(a.env,a.root.resolve(),a.origin)
