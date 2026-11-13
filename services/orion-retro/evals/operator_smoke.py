"""Exercise the deployed API, without ever loading a native core in the client."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid
import httpx
from websockets.asyncio.client import unix_connect


async def smoke(socket, token, game, seconds):
    headers={'Authorization':'Bearer '+token}
    client=httpx.Client(transport=httpx.HTTPTransport(uds=str(socket)),base_url='http://localhost',headers=headers,timeout=3)
    def request(path, body=None):
        response=client.get(path) if body is None else client.post(path,json=body)
        response.raise_for_status();return response.json()
    start=request('/sessions',dict(game_id=game,idempotency_key=str(uuid.uuid4())))
    sid=start['session_id']; base='/sessions/'+sid
    try:
        for _ in range(100):
            status=request(base)
            if status['state']=='running':break
            if status['state']=='failed':raise RuntimeError(status)
            await asyncio.sleep(.1)
        assert status['state']=='running',status
        async with unix_connect(str(socket),uri='ws://localhost'+base+'/input',additional_headers=headers) as ws:
            hello=json.loads(await ws.recv())
            grant=request(base+'/control',dict(connection_id=hello['connection_id'],port=0))
            first=last=None; started=time.monotonic(); seq=0
            while time.monotonic()-started < seconds:
                # Start enters the game; then alternating right/fly demonstrates actuation,
                # not competence, reward, or successful level completion.
                mask=8 if seq<2 else (128 | (256 if (seq//5)%2 else 0))
                await ws.send(json.dumps(dict(version=1,session_id=sid,generation=hello['generation'],
                    lease=grant['lease'],epoch=grant['epoch'],seq=seq,port=0,mask=mask)))
                ack=json.loads(await asyncio.wait_for(ws.recv(),2));assert 'applied' in ack,ack
                first=first or ack['applied']['frame'];last=ack['applied']['frame'];seq+=1
                await asyncio.sleep(.05)
        await asyncio.sleep(.15)
        status=request(base)
        assert status['state']=='paused',status
        assert last>first and status['video_frames']>0 and status['audio_samples']>0,status
        return dict(status='PASS',game_id=game,session_id=sid,applied_inputs=seq,
                    first_frame=first,last_frame=last,disconnect_state=status['state'],
                    identity=status.get('identity'),unverified=['browser_play','game_competence','P2_game_effects'])
    finally:
        request(base+'/stop',{})
        client.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',type=Path,default=Path.home()/'.local/share/orion/retro/service/ipc/retro.sock')
    p.add_argument('--token-file',required=True,type=Path)
    p.add_argument('--game',required=True)
    p.add_argument('--seconds',type=float,default=5)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if not 2<=args.seconds<=2400:p.error('seconds must be 2..2400')
    with args.output.open('x') as f:
        result=asyncio.run(smoke(args.socket,args.token_file.read_text().strip(),args.game,args.seconds))
        json.dump(result,f,indent=2);f.write('\n')
    print(result['status'])


if __name__=='__main__':main()
