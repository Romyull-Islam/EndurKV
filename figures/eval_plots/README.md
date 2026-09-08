# figures/eval_plots/

This directory holds the Wave-11 evaluation figures. **It is intentionally empty**
until real measurements have landed on disk.

## Why empty?

Earlier passes shipped three placeholder PNGs (`ppl_per_policy.png`,
`wave11_interim_ppl.png`, `wave11_interim_niah.png`) which only contained the
text "No Wave-11 PPL data found" / "No Wave-11 NIAH generations yet". They were
removed because they could be mistaken for actual results.

## How to (re)generate

After a complete Wave-11 eval run (240 PPL chunks + 240 NIAH stimuli), run:

    python eval_pipeline/score_ppl.py                  # -> ppl_per_policy.png
    python eval_pipeline/wave11_interim_plot.py        # -> wave11_interim_{ppl,niah}.png

These scripts will refuse to write a figure when no data is available, so an
empty directory after a run means the run did not complete — investigate
`phone-logs/wave11_eval_*/progress.log` rather than the figures.
