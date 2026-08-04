# Shared Resources & Job Scheduling
Sam Foreman
2025-07-15

<link rel="preconnect" href="https://fonts.googleapis.com">

- [🎯 Submitting a job](#dart-submitting-a-job)
  - [Interactive jobs](#interactive-jobs)
  - [Batch jobs](#batch-jobs)
- [Polaris Activity Webpage](#polaris-activity-webpage)
- [📊 Live Status of Polaris](#bar_chart-live-status-of-polaris)

> [!NOTE]
>
> ### 🧭 No account needed
>
> You do **not** need an HPC account to follow this page — it’s
> conceptual. Here you’ll learn how schedulers share a supercomputer
> across users and what a *job* is; the commands below are readable by
> anyone, whether or not you have access to a cluster. You’ll only need
> real cluster access for the large-scale, hands-on labs later in the
> course (e.g. distributed training in \[1.4\], the LLM lab in \[2.1\],
> and the AI-for-science examples in \[03.x\]).

Supercomputers contain many computer *nodes* and not every application
will use ALL of them. Therefore, we use programs called *schedulers*
that allow users to schedule a *job* based on how many nodes they need
for a specified time.

A *job* is defined by a user and requires these parameters to be
defined:

run-time  
How long will this *job* run? 5 minutes? 60 minutes? 6 hours?

number-of-nodes  
How many compute *nodes* does the application need to run? 5 nodes? 100
nodes? 1000 nodes?

> [!NOTE]
>
> ### 🏙️ Analogy: the scheduler as a maître d’
>
> A supercomputer is a huge restaurant and every user wants a table
> (nodes) for a certain length of time (walltime). You don’t just walk
> in and sit down — you tell the **maître d’** (the scheduler) your
> party size and how long you’ll stay, and it seats you when a suitable
> table frees up. Ask for a small table for a short time and you’re
> seated quickly; ask for the whole restaurant for all night and you’ll
> wait. That trade-off *is* queue time.

## 🎯 Submitting a job

You interact with the scheduler in one of two modes:

- **Interactive** — the scheduler gives you a shell *on the compute
  nodes* and you type commands live. Great for developing and debugging.
- **Batch** — you write a small script describing the job and hand it to
  the scheduler, which runs it unattended when resources free up and
  writes the output to a file. Great for long or many runs.

The two schedulers you’ll meet at ALCF/NERSC are **PBS Pro** (Polaris,
Sophia at ALCF) and **SLURM** (Perlmutter at NERSC). The concepts are
identical; only the command names differ.

> [!NOTE]
>
> ### 🖥️ Which machine will we use?
>
> The course’s **primary hands-on machine is NERSC Perlmutter**, which
> uses **SLURM** — so if you’re following along, reach for the SLURM
> commands below. The same ideas map directly onto **PBS** systems like
> ALCF’s Polaris (just swap `sbatch`↔`qsub`, `squeue`↔`qstat`, and so
> on), so the PBS tabs are here for reference too. The full PBS↔SLURM
> command cheat-sheet lives on the setup page.

### Interactive jobs

<div class="panel-tabset">

#### PBS (Polaris @ ALCF)

``` bash
# request 1 node for 30 minutes from the debug queue
qsub -I -A <project> -q debug -l select=1 -l walltime=00:30:00 \
     -l filesystems=home:eagle
```

`-I` = interactive, `-A` = the project/allocation to charge, `-q` =
queue, `-l select=N` = number of nodes, `-l walltime=HH:MM:SS` = time
limit. When the job starts you land on a compute node.

#### SLURM (Perlmutter @ NERSC)

``` bash
# request 1 GPU node for 30 minutes from the interactive queue
salloc --nodes 1 --qos interactive --time 00:30:00 \
       --constraint gpu --account <project>
```

`--nodes` = node count, `--qos` = queue/quality-of-service, `--time` =
limit, `--constraint gpu` = node type, `--account` = allocation.
`salloc` drops you onto the allocated node.

</div>

### Batch jobs

Write a job script, then submit it. The scheduler directives live in
comments at the top (`#PBS ...` or `#SBATCH ...`).

<div class="panel-tabset">

#### PBS (`job.sh`)

``` bash
#!/bin/bash -l
#PBS -A <project>
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -N my_first_job

cd "$PBS_O_WORKDIR"          # PBS starts you in $HOME; cd back to where you submitted
echo "Running on $(hostname)"
python3 my_script.py
```

``` bash
qsub job.sh          # submit -> prints a job id like 1234567.polaris
qstat -u $USER       # check status of your jobs
qdel 1234567         # cancel a job
```

#### SLURM (`job.sh`)

``` bash
#!/bin/bash -l
#SBATCH --account <project>
#SBATCH --qos regular
#SBATCH --nodes 2
#SBATCH --time 00:30:00
#SBATCH --constraint gpu
#SBATCH --job-name my_first_job

echo "Running on $(hostname)"
srun python3 my_script.py    # srun launches your program across the allocated nodes
```

``` bash
sbatch job.sh        # submit -> prints "Submitted batch job 1234567"
squeue --me          # check status of your jobs
scancel 1234567      # cancel a job
```

</div>

> [!TIP]
>
> ### 📤 Where does my output go?
>
> A batch job has no terminal, so its stdout/stderr are written to files
> in the submit directory:
>
> - **PBS** → `<job-name>.o<jobid>` (stdout) and `<job-name>.e<jobid>`
>   (stderr)
> - **SLURM** → `slurm-<jobid>.out` (both, by default)
>
> `tail -f slurm-1234567.out` lets you watch a running job’s output
> live.

> [!TIP]
>
> ### 🧠 Check your understanding
>
> Test yourself — think it through, then reveal the answer.
>
> **Q1.** You ask for 512 nodes for 12 hours and your colleague asks for
> 1 node for 10 minutes. Whose job is likely to start sooner, and why?
>
> > [!NOTE]
> >
> > ### Show answer
> >
> > Your colleague’s. Small, short requests are far easier for the
> > scheduler to fit into gaps between other jobs (“backfill”), so they
> > usually start almost immediately; a large, long request must wait
> > for a big block to free up.
>
> **Q2.** What’s the difference between `qsub -I` and `qsub job.sh`?
>
> > [!NOTE]
> >
> > ### Show answer
> >
> > `-I` requests an **interactive** session (a live shell on the
> > compute nodes); `qsub job.sh` submits a **batch** job that runs the
> > script unattended and writes output to a file.
>
> **Q3.** Your SLURM job finished but printed nothing to your terminal.
> Where do you look for its output?
>
> > [!NOTE]
> >
> > ### Show answer
> >
> > In `slurm-<jobid>.out` in the directory you ran `sbatch` from.

## Polaris Activity Webpage

- We have a page that shows all the current activity on Polaris. The top
  of the page shows a graphical representation of all nodes. Colors
  indicate a *job* running on that *node*. Below the graphic there is a
  table that lists *running* and *queued* jobs.
  - *running* refers to jobs running right now on computer nodes. If you
    hover on a colored node in the graphic, it will highlight all the
    nodes used by the same job AND the job in the *running* table below.
  - *queued* jobs are waiting for an opening in which to run.

## 📊 Live Status of Polaris

<div id="fig-polaris-status">

<iframe src="https://status.alcf.anl.gov/#/polaris" loading="lazy" height="720" width="100%" frameborder="0" align="center" allowfullscreen style="border:0">

</iframe>

Figure 1: See: <https://status.alcf.anl.gov/#/polaris>

</div>
