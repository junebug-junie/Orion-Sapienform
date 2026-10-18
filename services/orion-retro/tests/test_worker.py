"""Deterministic worker command ordering without loading a native core."""
import queue
from types import SimpleNamespace
from app import worker
from settings import Settings


def exercise(monkeypatch,tmp_path,scenario, *, crash=False, quota=268435456, initial=True, loops=20):
    commands=queue.Queue(); urgent=queue.Queue(); result=[]; steps=[]
    clock=SimpleNamespace(now=0., loops=0)
    class Stop:
        def is_set(self):return clock.loops>=loops
    def sleep(seconds):clock.now+=max(seconds,.001);clock.loops+=1
    monkeypatch.setattr(worker,'time',SimpleNamespace(monotonic=lambda:clock.now,sleep=sleep))
    class Core:
        identity={'test_only':True}; av=SimpleNamespace(timing=SimpleNamespace(fps=60))
        video_frames=0;audio_samples=0;frame_hash='fixture';buttons=(0,0)
        def __init__(self,*args):pass
        def step(self,*masks):
            if crash:raise RuntimeError('fixture_crash')
            steps.append(masks);self.video_frames+=1;self.audio_samples+=800
        def close(self):pass
    monkeypatch.setattr(worker,'CoreProbe',Core)
    def enqueue(op,rid,connection='',data=None):
        commands.put_nowait(dict(op=op,id=rid,connection=connection,data=data or {},received_at=clock.now))
    class Replies:
        def put_nowait(self,row):
            result.append(row);scenario(row,enqueue)
    if initial:
        enqueue('control','grant1','one',dict(port=0,action='acquire'))
        enqueue('control','grant2','two',dict(port=1,action='acquire'))
    worker.run(commands,urgent,Replies(),Stop(),Settings(artifacts=tmp_path,session_bytes=quota),
               {'path':tmp_path/'unused.nes','ports':[0,1],'game_id':'test'},'session','generation')
    return result,steps


def test_disconnect_does_not_reject_other_port_and_resume_is_idempotent(monkeypatch,tmp_path):
    def scenario(row,send):
        if row.get('id')=='grant2':
            grant=row['result']
            send('resume','duplicate-resume')
            send('input','p2','two',dict(port=1,lease=grant['lease'],epoch=grant['epoch'],seq=1,mask=128))
            send('disconnect','bye','one')
    rows,steps=exercise(monkeypatch,tmp_path,scenario)
    ack=next(r for r in rows if r.get('id')=='p2')
    assert ack['result']['mask']==128 and ack['result']['frame']>0
    assert (0,128) in steps
    assert not any(r['kind']=='fatal' for r in rows)


def test_malformed_token_does_not_kill_worker(monkeypatch,tmp_path):
    def scenario(row,send):
        if row.get('id')=='grant2':
            g=row['result'];data=dict(port=1,lease='é'*32,epoch=g['epoch'],seq=1,mask=128)
            send('input','bad','two',data)
            send('input','good','two',dict(data,lease=g['lease']))
    rows,_=exercise(monkeypatch,tmp_path,scenario)
    assert next(r for r in rows if r.get('id')=='bad')['error']=='invalid_or_expired_lease'
    assert 'result' in next(r for r in rows if r.get('id')=='good')
    assert not any(r['kind']=='fatal' for r in rows)


def test_native_exception_marks_worker_fatal(monkeypatch,tmp_path):
    rows,_=exercise(monkeypatch,tmp_path,lambda *args:None,crash=True)
    assert any(r['kind']=='fatal' and 'fixture_crash' in r['error'] for r in rows)


def test_evidence_failure_stops_before_actuation(monkeypatch,tmp_path):
    rows,steps=exercise(monkeypatch,tmp_path,lambda *args:None,quota=1)
    assert not steps
    assert any(r['kind']=='fatal' and 'evidence_quota' in r['error'] for r in rows)


def test_no_controller_pauses_after_bounded_start_grace(monkeypatch,tmp_path):
    rows,steps=exercise(monkeypatch,tmp_path,lambda *args:None,initial=False,loops=600)
    assert 290<=len(steps)<=302
    assert any(r['kind']=='status' and r['state']=='paused' for r in rows)
