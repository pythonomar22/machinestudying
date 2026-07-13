This is a cluster that we have of L40S GPUs. We have this, you can run *nvidia-smi* to see. 

There is a time limit, we reserved these GPUs for interactive jobs, since the cluster is very competitive. 

You need to be aware of how much time we have left, and implement a lot of checkpointing and stuff like that, jsut in case the time limit runs out before our GPU jobs finish, get it? If we have a training job running for past/longer than how long we reserve the GPUs for, then we will lose all progress. You must make sure that this does not happen. 

Be smart with long runs and GPUs. Like @CLAUDE.md says, make sure to run smoke runs and be sure things work before long runs. 

And whenever you use GPUs, make sure that it gets past init stages and similar things, that the intent of the user is captured and that things are going well. You also need to periodically check and monitor and poll the GPUs, log the slurm output (or whatever it's called) to make sure that the jobs are progressing correctly. That is, if we launch a job for 8 hours, I don't want you to have to wait until the end of the eight hours to figure out that we are doing something wrong, or that it errored in the first 5 minutes, get it? Periodically monitor and check the statuses of our GPU jobs.