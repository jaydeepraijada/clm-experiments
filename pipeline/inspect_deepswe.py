import modal, json
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1","numpy==2.1.3","huggingface_hub==0.26.2")
app = modal.App("clm-inspect")
@app.function(image=image, timeout=600)
def run():
    import torch
    from huggingface_hub import snapshot_download
    h = snapshot_download("Contrastive-LM/deepswe-clm-heads-8k", local_dir="/tmp/h")
    ck = torch.load(h+"/best_head.pt", map_location="cpu", weights_only=False)
    print("keys",[k for k in ck if k not in("state_head","action_head")])
    print("cfg",json.dumps(ck.get("cfg"),default=str)); print("logit",float(torch.as_tensor(ck["logit_scale"]))); print("epoch",ck.get("epoch"),"metrics",ck.get("metrics"))
    for f in ("config.json","verification.json","split_summary.json","README.md"):
        print("==",f); print(open(h+"/"+f).read()[:2600])
@app.local_entrypoint()
def main(): run.remote()
