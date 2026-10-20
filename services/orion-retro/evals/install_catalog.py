"""Import operator-owned installed kit ROMs, create a credential and local compose env."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets


def install(kit, root, env):
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    catalog=root/'catalog';catalog.mkdir(exist_ok=True,mode=0o700)
    (root/'evidence').mkdir(exist_ok=True,mode=0o700)
    (root/'ipc').mkdir(exist_ok=True,mode=0o700)
    rows=[]
    for game,title,path,ports in [('pad-lab','Pad Lab',kit,[0,1]),
        ('nova','Nova the Squirrel',kit/'games/nova-the-squirrel-v1.0.6a',[0]),
        ('spacegulls','Spacegulls',kit/'games/spacegulls-1.1',[0,1])]:
        if not (path/'manifest.json').exists():continue
        m=json.loads((path/'manifest.json').read_text());rom=(path/m['rom']).resolve(strict=True)
        if not rom.is_relative_to(path.resolve()):raise ValueError('ROM outside kit')
        data=rom.read_bytes();sha=hashlib.sha256(data).hexdigest()
        if sha!=m['rom_sha256']:raise ValueError('ROM digest mismatch')
        dest=catalog/(game+'.nes')
        if dest.exists() and dest.read_bytes()!=data:raise ValueError('refusing different ROM')
        dest.write_bytes(data)
        rows.append(dict(game_id=game,title=title,file=dest.name,rom_sha256=sha,ports=ports,
                         provenance=m.get('game',{'source':'original authored diagnostic'})))
    (catalog/'catalog.json').write_text(json.dumps(rows,indent=2)+'\n')
    token=root/'operator.token'
    if not token.exists():
        fd=os.open(token,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(48)+'\n')
    auth=root/'hub-auth';auth.mkdir(exist_ok=True,mode=0o700)
    for name in ('hub.token',):
        path=auth/name
        if not path.exists():
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(32)+'\n')
    keys=dict(RETRO_HUB_TOKEN_FILE=str(auth/'hub.token'),RETRO_ENABLED='true',RETRO_SOCKET_DIR=str(root/'ipc'),RETRO_UID=str(os.getuid()),RETRO_GID=str(os.getgid()),
              RETRO_CATALOG_DIR=str(catalog),RETRO_EVIDENCE_DIR=str(root/'evidence'),RETRO_TOKEN_FILE=str(token))
    # Preserve operator overrides; only populate missing/blank configuration.
    existing=env.read_text().splitlines() if env.exists() else []
    seen=set();out=[]
    for line in existing:
        key,sep,value=line.partition('=')
        if sep and key in keys:
            seen.add(key)
            if not value or (key=='RETRO_ENABLED' and value=='false'):line=key+'='+keys[key]
        out.append(line)
    out += [key+'='+value for key,value in keys.items() if key not in seen]
    env.write_text('\n'.join(out)+'\n')
    print(f'Installed {len(rows)} local catalog entries; credential retained privately.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit',type=Path,default=Path.home()/'.local/share/orion/retro')
    p.add_argument('--root',type=Path,default=Path.home()/'.local/share/orion/retro/service')
    p.add_argument('--env',type=Path,required=True)
    a=p.parse_args();install(a.kit.expanduser().resolve(),a.root.expanduser().resolve(),a.env)
