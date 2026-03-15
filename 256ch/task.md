# 2D convolutional Kernel optimization

This is an experiment to have the LLM optimize a 2D convolutional kernel as much as possible.

It focuses on higher channel 2D convolutions (from 128 onwards).

## Setup
You will have to perform the following tasks before you start. Work with the user for it.
1. Look at the current GPU configuration. For this, first check if there is a `gpu_config.md` file already available in this project. Otherwise, run `nvidia-smi` and save that into a file `gpu_config.md` for later context. If you see two GPUs, use only the first one. All the experiments will run on a single GPU.
2. Read the in-scope files: The repo is small. Read these files for context:
    - `main.cu` - benchmarking code
    - `kernels/1_naiveconv.cu` - Naive convolution. The baseline that we want to start with. This is read-only
    - `kernels/0_cutlass.cu` -- Optimized solution using cutlass. This can be used only for reference and is read-only. 
3. Find if the folder `automatic_kernels` exists and has data in it. You are only allowed to write kernels in this folder
    - If it doesn't exist, create it.
    - If it exists, look at the files inside. The experiments will have progressive order, with file name X_YYYY.cu where X represents the iteration number. You want to look at the latest ones for context.
4. Confirm and go: confirm setup looks good

Kick off the experimentation once you get a confirmation. Raise any concerns.

## Experimentation
Each experiment runs on a single GPU, as we have defined in `gpu_config.md`. To run experiments, you should run `make && ./engine` in the `build` directory.

**What you CAN do:**

- Add kernels to `automatic_kernels` directory, always treating each iteration as a new experiment and saving the file with an iteration number and a very short description of what changed. For example, if your new kernel improves the naive one on global memory coalescing, the file could be called `2_global_coalesce.cu`.
- Modify `main.cu` to add meaningful metrics. Although in the end running time will be the most important. Focus on square resolutions from 512x512 256 channels. This file will output the latest kernel time in the last 
- Move from NCHW layout to NHWC if that improves speed.
- Be creative. Tuning number of warps, etc with a 5x room for improvement doesn't make a lot of sense. Maybe there are new, creative ideas that you can try before that.

**What you CANNOT do:**

- Modify any file inside the directory `kernels` as those will be modified by the user and serve as basis.
- Modify the folder `thirdparty` unless you get permission from the user to add a new library.

**The goal is simple: get the lowest running time for different resolutions and channel counts**. Everything is fair game: change SMEM swizzling, vectorized access, warp tiling using WMMA, double buffering... The only constraint is that the code compiles, kernel runs without crashing, and the unit tests against CPU kernel pass.  

**Simplicity criterion:** All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 ms improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 ms improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run:** Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary to the console like this:
```
Unit tests passed.
CUTLASS 3x3 Conv2D - Time: 2.32 ms, 0.03 TFLOPS
1. Naive 3x3 Conv2D - Time: 33.54 ms, 0.00 TFLOPS
```
You can extract the information from the latest implementation. It will always be in the last line:
```
tail -n1 logfile | grep -oP 'Time: \K[0-9.]+(?= ms)'
```

## Logging results

When an experiment is done, log it to `256ch/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:
```
commit  experiment_number   time    status  description
```

1. git commit hash (short, 7 chars)
2. experiment_number
3. Time in ms
4. status: `keep`, `discard` or `crash`
5. Short text description of what this experiment tried

```
commit experiment_number   time    status  description
a1b2c3d 1   33.56   keep    Naive convolution
```

## The experiment loop

LOOP FOREVER:
1. Look at latest implementations in `automatic_kernels`
2. Write a new kernel in `automatic_kernels` with an experimental idea
3. git commit
4. Compile the codebase in the `build` directory: `cd build && make > compile.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Make sure there are no compilation errors with `tail -n 2 compile.log`. It should have arrived to 100% with something like `[100%] Built target cutlass_example`. If the compilation crashed, read the stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
6. Run the experiment `./engine > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
7. Read out the result of the new kernel: `tail -n1 run.log | grep -oP 'Time: \K[0-9.]+(?= ms)'`
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If time improved (lower), you "advance" the branch, keeping the git commit
10. If time is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes:** If a run crashes (compilation error, OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP:** Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~1 minute then you can run approx 12/hour, for a total of about 500 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!