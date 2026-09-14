

- reset staleness

- method to unlock n blocks on backbone
- method to set lr warmup
    - warmup
    - read `scheduler.get_last_lr()[0] * ratio`
- ratio per stage
    0.95^()




Here's the plan, in the order you'd actually build it:

**1. Trigger metric**
Reuse the existing stale-counter mechanism (same `min_delta`) with its own, shorter `unfreeze_patience` — separate from `stopping_patience`. No slope/regression needed to start; revisit only if logs show it firing on noise.

**2. Aggregation**
Decide explicitly: does unfreeze require `min(stale_all_three) >= unfreeze_patience` (mirrors stopping, but risks never firing given Budding's noise) or a looser any/majority rule (unlocks sooner, risks moving on before a class is ready)? This is still open — pick one before wiring the rest.

**3. Cooldown / anti-cascade**
On every trigger fire: reset all three stale counters to 0. This alone prevents immediately re-triggering next epoch — no extra boolean/latch needed.

**4. Granularity + budget check**
Define stages as block-groups, not individual layers. Given warmup(3)+cosine(T≈7) eating ~10 epochs before first full cycle, and patience typically firing ~19–21, work out concretely how many stages fit — this bounds `max_layers_to_reach` (or better, `max_stages`) before you pick a number arbitrarily.

**5. Per-block state**
Each block needs: `unlock_epoch` (int, set the epoch its trigger fires) and a small `local_warmup_len` (can be a fixed constant, e.g. same 3 as global warmup). Store both — this is the new checkpoint field.

**6. LR mechanics — single scheduler, manual overlay**
- The existing global `SequentialLR` keeps tracking whatever it already tracks today — no `add_param_group`, no second scheduler instance.
- Each epoch, for every currently-unlocked block: if `epoch - unlock_epoch < local_warmup_len`, set its LR by hand via a small linear/warmup function of that local counter; otherwise set `lr = scheduler.get_last_lr()[0] * depth_ratio_i`.
- `depth_ratio_i`: gentle multiplicative decay per depth (start around 0.9–0.95×, not `10^-depth`) — hand-picked constant for now, flag as an Optuna candidate later once the unfreezing approach itself is validated.

**7. Checkpoint/resume**
On resume, recompute each unlocked block's phase from stored `unlock_epoch` vs. current epoch — same logic as the forward pass, just replayed. No new scheduler state to desync, since there's still only one scheduler object.

**8. Build order**
a. Block-level unfreeze method on `backbone` (mechanical, low-risk, do first).
b. Trigger + cooldown logic reusing existing stale counters.
c. Manual per-block LR overlay function.
d. Checkpoint field for `unlock_epoch`/resume replay.
e. Run with `max_stages` conservative (probably 2, per step 4) before tuning further.

**Still open, in priority order:** aggregation rule (step 2), and how many stages your epoch budget actually supports (step 4) — both are judgment calls, not mechanical ones, worth deciding deliberately rather than defaulting.
