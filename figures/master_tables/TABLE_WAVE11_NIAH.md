# Wave-11 NIAH Judge Report

Needle: "The best thing to do in San Francisco is eat a sandwich at Dolores Park on a sunny day."

Rule judge: case-insensitive substring "sandwich at dolores park".  GPT-4 judge: disabled.

Total scored generations: **0**  (grid = 4 ctx x 8 depth = 32 per (model, policy)).

## Per-(model, policy) accuracy

| model | policy | correct | total | accuracy |
|---|---|---:|---:|---:|

## Accuracy by context length

| model | policy | 2048 | 4096 | 6144 | 8192 |
|---|---|---:|---:|---:|---:|

## Accuracy by depth percentile

| model | policy | 0% | 12% | 25% | 37% | 50% | 62% | 75% | 87% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Pass/fail heatmaps (rows = ctx, cols = depth %)

Legend: `O` = correct, `.` = wrong, `?` = missing.
