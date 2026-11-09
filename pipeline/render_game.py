"""Draw recorded T-Rex runs side by side, from real frames saved by run_trex.py. Original simple graphics.

    modal run render_game.py --left authors/shield --right authors/noshield --seed 0 --name authors_shield_vs_noshield
"""
import gzip
import json
import os
import subprocess

import modal

image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "fonts-dejavu-core").pip_install("pillow==11.0.0"))
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
app = modal.App("clm-render")
W, H, GROUND = 600, 150, 137
INK, PAPER, ORANGE, BLUE, GREY = (22, 33, 58), (250, 248, 243), (184, 74, 20), (42, 98, 168), (150, 155, 165)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def runner(d, y, ducking, crashed, t):
    """A plain round-bodied runner, drawn from shapes. y is the top of a 47 px tall hitbox."""
    x, c = 50, ORANGE if crashed else INK
    if ducking:
        d.rounded_rectangle([x, y + 22, x + 52, y + 42], 8, fill=c)
        d.ellipse([x + 42, y + 18, x + 60, y + 34], fill=c)
        d.ellipse([x + 52, y + 22, x + 56, y + 26], fill=PAPER)
    else:
        d.rounded_rectangle([x + 4, y + 14, x + 32, y + 38], 9, fill=c)
        d.ellipse([x + 20, y, x + 44, y + 20], fill=c)
        d.ellipse([x + 34, y + 5, x + 38, y + 9], fill=PAPER)
        d.polygon([(x + 4, y + 20), (x - 8, y + 30), (x + 6, y + 32)], fill=c)
    step = (t // 4) % 2 if y >= 90 else 0
    d.rectangle([x + 10, y + 38, x + 15, y + 47 - 4 * step], fill=c)
    d.rectangle([x + 22, y + 38, x + 27, y + 43 + 4 * step], fill=c)


def obstacle(d, kind, x, y, w, frame):
    if "bird" in kind.lower() or "ptero" in kind.lower():
        up = frame % 2 == 0
        d.polygon([(x, y + 18), (x + 18, y + 12), (x + 26, y + (0 if up else 34)), (x + 32, y + 12), (x + w, y + 20), (x + 16, y + 26)], fill=BLUE)
        return
    big = "large" in kind.lower()
    top, each = (GROUND - 50 if big else GROUND - 35), (25 if big else 17)
    for i in range(max(1, int(round(w / each)))):
        cx = x + i * each + each // 2
        d.rounded_rectangle([cx - 4, top, cx + 4, GROUND], 3, fill=(46, 125, 80))
        d.rectangle([cx - 10, top + 12, cx + 10, top + 16], fill=(46, 125, 80))


def panel(frame, title, sub, t, f, fb):
    from PIL import Image, ImageDraw
    y, duck, crashed, score, speed, obs = frame
    im = Image.new("RGB", (W, H + 48), PAPER)
    d = ImageDraw.Draw(im)
    d.text((10, 6), title, font=fb, fill=INK)
    d.text((10, 27), sub, font=f, fill=(86, 96, 110))
    d.text((W - 118, 8), f"score {score:>5}", font=fb, fill=INK)
    o = 48
    from PIL import Image as I
    g = I.new("RGB", (W, H), PAPER)
    gd = ImageDraw.Draw(g)
    gd.line([(0, GROUND), (W, GROUND)], fill=GREY, width=2)
    for k, x, oy, w, fr in obs:
        obstacle(gd, k, int(x), int(oy), int(w), fr)
    runner(gd, int(y), bool(duck), bool(crashed), t)
    if crashed:
        gd.text((W // 2 - 60, 40), "CRASHED", font=fb, fill=ORANGE)
    im.paste(g, (0, o))
    return im


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=16000, timeout=1800)
def render(left, right, seed, name, seconds=40, ltitle="", rtitle="", lsub="", rsub=""):
    from PIL import Image, ImageFont
    f, fb = ImageFont.truetype(FONT, 13), ImageFont.truetype(BOLD, 15)
    vol.reload()
    A = json.load(gzip.open(f"/data/trex/{left}.json.gz", "rt"))
    B = json.load(gzip.open(f"/data/trex/{right}.json.gz", "rt"))
    print("runs", [(r["seed"], len(r["frames"])) for r in A], [(r["seed"], len(r["frames"])) for r in B], flush=True)
    pick = lambda R: max((r for r in R if r["seed"] == seed), key=lambda r: len(r["frames"]))
    a, b = pick(A)["frames"], pick(B)["frames"]
    n = min(len(a), len(b), int(seconds * 60))
    out = f"/tmp/{name}"
    os.makedirs(out, exist_ok=True)
    for i, t in enumerate(range(0, n, 2)):  # 30 fps
        pa, pb = panel(a[t], ltitle or left, lsub, t, f, fb), panel(b[t], rtitle or right, rsub, t, f, fb)
        im = Image.new("RGB", (W, 2 * (H + 48) + 6), (220, 216, 205))
        im.paste(pa, (0, 0)); im.paste(pb, (0, H + 54))
        im.save(f"{out}/{i:05d}.png")
    os.makedirs("/data/renders", exist_ok=True)
    mp4, gif = f"/data/renders/{name}.mp4", f"/data/renders/{name}.gif"
    subprocess.run(f"ffmpeg -y -loglevel error -framerate 30 -i {out}/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 -vf scale=1200:-2:flags=neighbor {mp4}", shell=True, check=True)
    subprocess.run(f"ffmpeg -y -loglevel error -framerate 30 -i {out}/%05d.png -vf 'fps=15,split[s0][s1];[s0]palettegen=max_colors=32[p];[s1][p]paletteuse=dither=none' {gif}", shell=True, check=True)
    subprocess.run(f"cp {out}/00300.png /data/renders/{name}_sample.png", shell=True)
    vol.commit()
    print("sizes", os.path.getsize(mp4), os.path.getsize(gif), "frames", n // 2, flush=True)


@app.local_entrypoint()
def main(left: str, right: str, seed: int = 0, name: str = "game", seconds: int = 40, ltitle: str = "", rtitle: str = "", lsub: str = "", rsub: str = ""):
    render.remote(left, right, seed, name, seconds, ltitle, rtitle, lsub, rsub)
