# SEA-SSAIE 2027 benchmark kit

Copy-paste steps for running the benchmark on the three machines. Every script can be stopped
with Ctrl+C and re-run; it skips work that is already finished.

**What runs where**

| Machine | Runs | Approx. time |
|---|---|---|
| Laptop (WSL) | Item prep, then full quality run (9 models x 220 requests) + concurrency | 4-5 h |
| VPS | Speed run (8 models x 25 requests) + concurrency + all Gemini calls | 2-3 h |
| Homelab LXC | Speed run (8 models x 25 requests) + concurrency | 2-3 h |

The three machines can run at the same time. Total model downloads: about 16 GB per server, 21 GB on the laptop.

---

## Step 0 - Laptop power settings (Windows)

The run takes hours, and a sleeping laptop stops it.

1. Settings > System > Power: when plugged in, **sleep = Never**; lid close action = **Do nothing**.
2. Keep the laptop plugged in, and keep the WSL terminal window open (closing it can stop WSL).

## Step 1 - Laptop: set up and build the item set (about 10 min)

In WSL:

```bash
cd "/mnt/c/Users/chada/Documents/202610 SSAIE Project/ssaie-bench"
bash setup.sh laptop
```

The last lines should say `Computer science items found: 410` and `Wrote .../items.jsonl: 200 items`.

## Step 2 - Homelab: create the throwaway LXC (about 10 min)

On the Proxmox host (`root@abinguna`):

```bash
pct list                                   # pick an unused ID; 950 is used below
pveam update
pveam available --section system | grep ubuntu-24.04
# download the template name shown by the previous command, e.g.:
pveam download local ubuntu-24.04-standard_24.04-2_amd64.tar.zst

pct create 950 local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst \
  --hostname ssaie-bench --cores 4 --memory 4096 --swap 0 \
  --rootfs local-lvm:25 --net0 name=eth0,bridge=vmbr0,ip=dhcp \
  --unprivileged 1 --features nesting=1 --start 1
```

If your storage is not called `local-lvm`, use the name shown by `pvesm status`.
During the run, pause any heavy containers or VMs on this host so they don't skew the timings.

## Step 3 - Copy the kit to the servers (from WSL)

```bash
cd "/mnt/c/Users/chada/Documents/202610 SSAIE Project"
tar czf /tmp/ssaie-bench.tgz --exclude=ssaie-bench/results ssaie-bench

# VPS (use your Tailscale name or IP)
scp /tmp/ssaie-bench.tgz chada@<VPS>:~/
ssh chada@<VPS> 'tar xzf ~/ssaie-bench.tgz'

# Homelab: to the Proxmox host, then into the LXC
scp /tmp/ssaie-bench.tgz root@abinguna:/root/
ssh root@abinguna 'pct push 950 /root/ssaie-bench.tgz /root/ssaie-bench.tgz && pct exec 950 -- tar xzf /root/ssaie-bench.tgz -C /root'
```

## Step 4 - Start all three runs

Each runs inside `tmux`, so it keeps going if your SSH connection drops.
Detach with `Ctrl+B` then `D`; re-attach with `tmux attach -t bench`.

**Laptop (WSL):**
```bash
cd "/mnt/c/Users/chada/Documents/202610 SSAIE Project/ssaie-bench"
tmux new -s bench 'python3 bench.py --tier laptop 2>&1 | tee -a results/laptop_console.log'
```

**VPS** (pause your other services first if you can):
```bash
ssh chada@<VPS>
cd ~/ssaie-bench && bash setup.sh vps
tmux new -s bench 'python3 bench.py --tier vps 2>&1 | tee -a results/vps_console.log'
```

**Homelab LXC:**
```bash
ssh root@abinguna
pct enter 950
cd /root/ssaie-bench && bash setup.sh homelab
tmux new -s bench 'python3 bench.py --tier homelab 2>&1 | tee -a results/homelab_console.log'
```

A model that does not fit in memory (most likely `gemma3:4b` on the servers) is logged as
`FAILED TO LOAD` and skipped. That is an expected result, not an error to fix.

## Step 5 - Gemini answers and items (VPS, can run alongside Step 4)

In a second tmux window on the VPS:
```bash
cd ~/ssaie-bench
tmux new -s gemini 'python3 frontier.py answer; python3 frontier.py generate'
```
If it prints "Free daily quota used up", re-run the same command later (or the next day); it resumes.
To force a different Gemini model: `GEMINI_MODEL=gemini-2.5-flash-lite python3 frontier.py answer`.

## Step 6 - Gemini judging (VPS, after the laptop finishes)

```bash
# from WSL: send the laptop's results to the VPS
scp "/mnt/c/Users/chada/Documents/202610 SSAIE Project/ssaie-bench/results/laptop/quality.jsonl" \
    chada@<VPS>:~/ssaie-bench/results/laptop/
# then on the VPS:
cd ~/ssaie-bench && python3 frontier.py judge
```
(Create the folder first if needed: `ssh chada@<VPS> 'mkdir -p ~/ssaie-bench/results/laptop'`.)

## Step 7 - Bring all results back to the laptop folder (from WSL)

```bash
cd "/mnt/c/Users/chada/Documents/202610 SSAIE Project/ssaie-bench/results"
scp -r chada@<VPS>:~/ssaie-bench/results/vps chada@<VPS>:~/ssaie-bench/results/frontier .
ssh root@abinguna 'pct exec 950 -- tar czf /root/res.tgz -C /root/ssaie-bench/results homelab && pct pull 950 /root/res.tgz /root/res-homelab.tgz'
scp root@abinguna:/root/res-homelab.tgz . && tar xzf res-homelab.tgz && rm res-homelab.tgz
```

Then tell Claude the results are in. Claude runs the analysis, prepares your 40-item rating sheet, and
drafts the paper.

## Cleanup (after the paper is done)

```bash
ssh root@abinguna 'pct stop 950 && pct destroy 950'      # homelab LXC
rm -rf ~/ssaie-models                                     # downloaded models, on VPS and laptop
sudo systemctl enable --now ollama                        # only if you used Ollama as a service before
```

## Files

| File | Purpose |
|---|---|
| `config.json` | Models, decoding settings, Gemini preferences |
| `prepare_items.py` | Builds `data/items.jsonl` (200 MMLU-Pro CS items, seed 2027) |
| `data/topics.json` | 20 item-writing topics with Bloom levels |
| `bench.py` | Local benchmark: answers, generated items, timings, memory, concurrency |
| `frontier.py` | Gemini baseline answers, generated items, and judging |
| `results/<tier>/` | Raw results (JSONL) + `sysinfo.json` per machine |
