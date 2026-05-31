from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch, torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.data import iter_batches, load_fineweb_edu_tokens, load_text_file_tokens, load_tinystories_tokens, shard_token_ids
from orion.substrate.experiments.hyperbolic_gpt.manifold_metrics import summarize_manifold_batch
from orion.substrate.experiments.hyperbolic_gpt.moc_patch import apply_moc, model_geometry_diagnostics
from orion.substrate.experiments.hyperbolic_gpt.model_v2 import HyperbolicGPTV2


def ddp():
    if all(k in os.environ for k in ("RANK","WORLD_SIZE","LOCAL_RANK")):
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    return False, 0, 1, 0

def raw(m): return m.module if isinstance(m, DDP) else m

def say(rank, msg):
    if rank == 0: print(msg, flush=True)

def split(ids):
    n = max(1, int(len(ids) * 0.05)); return ids[:-n], ids[-n:]

def rmean(x, device, is_ddp):
    t = torch.tensor([x], dtype=torch.float64, device=device)
    if is_ddp: dist.all_reduce(t); t /= dist.get_world_size()
    return float(t.item())

def ids(args, rank):
    if args.dataset == "tinystories":
        try: return load_tinystories_tokens(args.max_docs, args.max_tokens)
        except Exception as e:
            if not args.text_path: raise RuntimeError("TinyStories unavailable; pass --text_path") from e
            say(rank, f"TinyStories failed ({e}); using text fallback"); return load_text_file_tokens(args.text_path, args.max_tokens)
    if args.dataset == "fineweb_edu":
        try: return load_fineweb_edu_tokens(args.max_docs, args.max_tokens, args.fineweb_edu_name)
        except Exception as e:
            if not args.text_path: raise RuntimeError("FineWeb-Edu unavailable; pass --text_path") from e
            say(rank, f"FineWeb-Edu failed ({e}); using text fallback"); return load_text_file_tokens(args.text_path, args.max_tokens)
    if not args.text_path: raise ValueError("--text_path required for --dataset text")
    return load_text_file_tokens(args.text_path, args.max_tokens)

@torch.no_grad()
def eval_all(model, ev_ids, cfg, device, bs):
    model.eval(); rows = []
    for x, y in iter_batches(ev_ids, cfg.block_size, bs, device):
        logits, loss, aux = model(x, y, return_aux=True)
        rows.append(summarize_manifold_batch(logits=logits, hidden=aux["hidden"], tangent=aux.get("tangent"), loss_ce=aux.get("loss_ce", loss.detach()), gap_epsilon=cfg.margin_gap_epsilon))
        if len(rows) >= 20: break
    model.train(); d = max(len(rows), 1)
    return {"loss_ce":sum(r.loss_ce for r in rows)/d, "gap_frac":sum(r.gap_frac for r in rows)/d, "margin_mean":sum(r.margin_mean for r in rows)/d, "entropy_mean":sum(r.entropy_mean for r in rows)/d, "fisher_trace_proxy":sum(r.fisher_trace_proxy for r in rows)/d, "tangent_energy":sum(r.tangent_energy for r in rows)/d}

def parse():
    p=argparse.ArgumentParser(description="Train v3 MoC: v2 diagnostics plus per-head geometry")
    p.add_argument("--dataset", choices=("tinystories","fineweb_edu","text"), required=True); p.add_argument("--fineweb_edu_name", default="sample-10BT"); p.add_argument("--text_path", default=None); p.add_argument("--out_dir", default="./runs/hypgpt_v3_moc")
    p.add_argument("--max_steps", type=int, default=1000); p.add_argument("--max_docs", type=int, default=None); p.add_argument("--max_tokens", type=int, default=None); p.add_argument("--batch_size", type=int, default=4); p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--block_size", type=int, default=512); p.add_argument("--n_layer", type=int, default=12); p.add_argument("--n_head", type=int, default=12); p.add_argument("--n_embd", type=int, default=768); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--semantic_adapter_rank", type=int, default=128); p.add_argument("--margin_gap_loss_weight", type=float, default=0.001); p.add_argument("--margin_gap_epsilon", type=float, default=0.5)
    p.add_argument("--curvature_mode", choices=("global","per_head"), default="per_head"); p.add_argument("--geo_lambda_mode", choices=("global","per_head"), default="per_head"); p.add_argument("--moc_curvature_jitter", type=float, default=0.05); p.add_argument("--moc_lambda_jitter", type=float, default=0.05)
    p.add_argument("--eval_interval", type=int, default=500); p.add_argument("--save_interval", type=int, default=2500); p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()

def main():
    a=parse(); is_ddp, rank, world, local = ddp()
    if is_ddp: dist.init_process_group("nccl"); torch.cuda.set_device(local); device=torch.device("cuda", local); print(f"[ddp] rank={rank} local_rank={local} world={world} device={device}", flush=True)
    else: device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=HyperbolicGPTConfig(block_size=a.block_size,n_layer=a.n_layer,n_head=a.n_head,n_embd=a.n_embd,semantic_adapter_rank=a.semantic_adapter_rank,margin_gap_loss_weight=a.margin_gap_loss_weight,margin_gap_epsilon=a.margin_gap_epsilon,curvature_mode=a.curvature_mode,geo_lambda_mode=a.geo_lambda_mode,moc_curvature_jitter=a.moc_curvature_jitter,moc_lambda_jitter=a.moc_lambda_jitter)
    all_ids=ids(a, rank); tr, ev = split(all_ids)
    if is_ddp: tr=shard_token_ids(tr,rank,world); ev=shard_token_ids(ev,rank,world)
    model=HyperbolicGPTV2(cfg).to(device); apply_moc(model,cfg)
    if is_ddp: model=DDP(model, device_ids=[local], output_device=local)
    opt=torch.optim.AdamW(model.parameters(), lr=a.lr); scaler=GradScaler(enabled=a.amp and device.type=="cuda"); batches=iter_batches(tr,cfg.block_size,a.batch_size,device); opt.zero_grad(set_to_none=True)
    out=Path(a.out_dir); step=micro=n=0; loss_sum=0.0
    while step<a.max_steps:
        try: x,y=next(batches)
        except StopIteration: batches=iter_batches(tr,cfg.block_size,a.batch_size,device); x,y=next(batches)
        with autocast(enabled=a.amp and device.type=="cuda"):
            _,loss,_=model(x,y); loss=loss/a.grad_accum
        scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward(); loss_sum+=float(loss.item())*a.grad_accum; n+=1; micro+=1
        if micro%a.grad_accum: continue
        if scaler.is_enabled(): scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if scaler.is_enabled(): scaler.step(opt); scaler.update()
        else: opt.step()
        opt.zero_grad(set_to_none=True); step+=1
        if step%10==0: say(rank,f"step {step} loss {loss_sum/max(n,1):.4f}"); loss_sum=0.0; n=0
        if step%a.eval_interval==0:
            m={k:rmean(v,device,is_ddp) for k,v in eval_all(raw(model),ev,cfg,device,a.batch_size).items()}; g=model_geometry_diagnostics(model)
            say(rank,f"eval step {step} loss_ce {m['loss_ce']:.4f} gap@{cfg.margin_gap_epsilon:.2f} {m['gap_frac']:.3f} margin {m['margin_mean']:.3f} entropy {m['entropy_mean']:.3f} fisher_proxy {m['fisher_trace_proxy']:.3f} tangent {m['tangent_energy']:.6f} cμ {g.get('moc_curvature_mean',0):.4f} cσ {g.get('moc_curvature_std',0):.4f} λμ {g.get('moc_geo_lambda_mean',0):.4f} λσ {g.get('moc_geo_lambda_std',0):.4f}")
        if step%a.save_interval==0:
            if rank==0: out.mkdir(parents=True,exist_ok=True); torch.save({"model":raw(model).state_dict(),"config":cfg.to_dict()},out/"model.pt"); cfg.save_json(out/"config.json"); (out/"meta.json").write_text(json.dumps({"model_class":"HyperbolicGPTV3MoC","tokenizer":"gpt2"}),encoding="utf-8"); print(f"saved checkpoint to {out}",flush=True)
    if rank==0: out.mkdir(parents=True,exist_ok=True); torch.save({"model":raw(model).state_dict(),"config":cfg.to_dict()},out/"model.pt"); cfg.save_json(out/"config.json"); (out/"meta.json").write_text(json.dumps({"model_class":"HyperbolicGPTV3MoC","tokenizer":"gpt2"}),encoding="utf-8"); print("training complete",flush=True)
    if is_ddp: dist.barrier(); dist.destroy_process_group()

if __name__=="__main__": main()
