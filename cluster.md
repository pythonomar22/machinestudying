TLDR:

# Cluster workflow (referenced by CLAUDE.md; written from observed practice)

## Layout

- Sessions run **on the allocated compute node** (hostname = the slurm node,
  e.g. `slurm0us-a3nodeset-2`) inside an interactive slurm job.
  `squeue -u $USER -o "%.10i %.12L %b"` shows job id, TIME_LEFT, and gpus.
- a3 partition = 8×H100-80GB nodes, 5-day max. Typical grant: 4 GPUs for 24h.
- **Multiple concurrent allocations are possible** (e.g. 4 GPUs on two nodes
  = 8 total): `salloc -w <node> --gres=gpu:4 --time=24:00:00` from the login
  node. The salloc job lives only as long as its shell — keep it inside a
  tmux session and DO NOT close it. Inside the salloc shell slurm shows a
  constrained, RENUMBERED GPU view (0..N-1); via ssh to the node you see
  all 8 physical GPUs — match pci.bus_id to find which physical indices are
  yours (e.g. job 20736's 0A/85/8A/8B = physical 2,5,6,7 on a3nodeset-3)
  and pin CUDA_VISIBLE_DEVICES to the physical ids when launching via ssh.
- **Nodes are SHARED and shells are NOT cgroup-constrained**: nvidia-smi
  shows all 8 GPUs including other users' processes, and
  CUDA_VISIBLE_DEVICES is unset. Our GPUs are whichever are free —
  verify with `nvidia-smi --query-gpu=index,memory.used --format=csv` and
  ALWAYS set CUDA_VISIBLE_DEVICES explicitly per process.
  `scripts/fade/launch_probe.sh` enforces a <1GiB-used check + refuses
  duplicate launches per run dir; FADE_GPUS overrides its default 0,5,6,7.

## Batch jobs (preferred over salloc since 2026-07-03)

- `sbatch scripts/fade/batch/<x>.sbatch` from the login/IDE node; scripts
  set partition a3, gpus-per-node, time, and log to `logs/slurm/%x-%j.out`.
  Inside a batch job slurm cgroup-isolates the allocated GPUs and sets
  CUDA_VISIBLE_DEVICES (renumbered 0..N-1) — do NOT override it.
- Slurm also sets ROCR_VISIBLE_DEVICES; ray/verl refuses to start with both
  present — `_run_verl.sh` unsets it (any new ray-based script must too).
- Per-user GPU quota (QOSMaxGRESPerUser) counts interactive + batch jobs
  together; zombie pending sallocs block batch jobs — `scancel` them.
- Long jobs must still be resumable (probe chunks, ckpt/10): batch jobs die
  at --time like sallocs do. Everything here is safe to resubmit as-is.

## Rules of thumb

1. **Check TIME_LEFT before launching anything long**; everything must be
   chunk-resumable (probe chunks, 1.7B training ckpt every 10 steps) so
   losing the allocation loses minutes, not hours. Verify a run's FIRST
   checkpoint actually lands (save cadence × step time vs TIME_LEFT) —
   a run that dies before its first save leaves nothing.
2. Long jobs: `nohup ... > logs/x.log 2>&1 &` + a monitor/heartbeat; verify
   the job gets PAST INIT and is actually generating/stepping before
   walking away (CLAUDE.md rule).
3. /home is NFS (slow small-file IO; uv needs UV_LINK_MODE=copy; killed
   processes leave .nfs* silly-rename files that clear on handle release).
4. 104 cores, 1.8TB RAM per node; HF cache in ~/.cache/huggingface.
5. System CUDA: /usr/local/cuda-12.8 (no 12.6 despite older script
   defaults). Driver 580.x.
6. Throughput planning numbers: `analysis/throughput_memo.md`
   (Qwen3-1.7B decode is KV-bandwidth-bound ≈3.1k tok/s/GPU at ~10k ctx).

7. Killing GPU squatters: vLLM engine workers rename their process title
   (e.g. "VLLM::EngineCore") — cmdline pkill patterns miss them. Always
   list PIDs via `nvidia-smi --query-compute-apps=gpu_uuid,pid` for the
   target GPU and `kill -KILL` by PID. And never pkill a pattern contained
   in your own remote command line (it kills your ssh shell first).



GCP Manual at Sakana AI

Author(s): Yujin Tang Kou Misaki Ciaran Regan Yingtao Tian Koshi Eguchi
Update date: Jul 1, 2026


✅ Overview
This document guides new GCP users through setting up their accounts and configuring remote access. After completing this manual, you will be able to connect to a virtual machine on the Google Cloud Platform via SSH.


📚 Table of Contents
There are three key sections:
Instructions – Covers everything from SSH-ing into GCP to submitting a Slurm job. All essential commands and steps are documented here.
Cluster & Slurm Tips – Provides helpful advice for working with the cluster. Not mandatory, but worth reading.
Troubleshooting – Lists real issues users have faced. Consult this section whenever you run into problems on GCP.



👀 Before you read
🚧 [IMPORTANT] Regular host maintenance on GCP and related tips
GPU VMs in GCP automatically restart if we do not reboot them ourselves (This automatic reboot is called host maintenance.)
To avoid the host maintenance, we periodically reboot the compute instances with GPUs (but not the login node) every 10–11 days.
Please join the cluster_users@sakana.ai (https://groups.google.com/a/sakana.ai/g/cluster-users) Google Group to get a Google Calendar invitation for our maintenance schedule.
Please be prepared for the reboot by checkpointing your jobs appropriately.
While GCP login nodes are generally not rebooted during maintenance, they may be restarted on rare occasions.

🔥 Troubleshooting
There is a Troubleshooting Section. If you cannot resolve issues, please check this section.
GCP Q&A: cluster_users@sakana.ai group is also a good place to ask questions about GCP.  Join the cluster_users@sakana.ai group and post your question there. 
Once the problem is fixed, please add both the issue and its solution to this Troubleshooting Section.



📝 1. Instructions
Currently, we have two GCP clusters:
Name
Region
# of GPU Nodes 
Introduced in
EU Cluster
EU 🇪🇺
16 (w/ 8 H100 GPUs each)
Aug, 2024
US Cluster
US 🇺🇸
14 (w/ 8 H100 GPUs each)
Dec, 2024


Both clusters run Slurm and submit jobs to compute nodes via the corresponding login node (slurm0-login-001 in Europe region and slurm0us-login-001 in the US region). Please note that you cannot submit jobs from a separate region.


🚧 Note: Why We Separate File Stores? 🚧
The GCP clusters in the EU and US use separate FileStores, meaning that data is not shared between the EU and US regions!
While it was technically possible to use a shared FileStore for both EU and US regions, we chose not to do so due to the risk of severe performance degradation caused by cross-region I/O operations. As a result, if you need to, for example, save the progress of a program running in the US and resume it in the EU, please refer to How to share data across the two clusters for guidance on how to share data between regions. That said, sharing data between the US and EU is expected to be time-consuming, so we strongly recommend running programs within a single cluster whenever possible.





🚀 1.1. gcloud CLI Setup
1.1.1. gcloud installation
To set up, you need the gcloud command-line tool. Please follow the installation guide here.
After installation, run:
gcloud init
When prompted for a project, select school-of-fish-412803.

First Time Access!!
If this is your first time accessing the cluster and you cannot select the project, please join cluster_users@sakana.ai (https://groups.google.com/a/sakana.ai/g/cluster-users) group.



 1.2. Slurm Compute Nodes Setup

To get started, you need to log in to the “login node” and see if the compute resources are available. The login and compute nodes share the /home folder by NFS, so you can set up your environment inside the login node as well.

For more details and commands, please refer to the official documentation or an LLM.

If you just want to use VS Code with GPU resources, skip ahead to VS Code Remote SSH and Development on Slurm Compute Node via VS Code.

Also, please note that we impose 8944MB RAM limit per CPU. Please increase the number of CPUs if the memory usage of your job is high.
By default, 26 CPUs and 232GB RAM per GPU are assigned to your job.
 1.2.1. SSH to the Login Node
First, we need to login to the “login node” which is used for submitting jobs to compute nodes:
🇪🇺 The external IP address of the login node in Europe: 34.91.238.94
🇺🇸 The external IP address of the login node in the US: 34.136.207.119

Please add the following ssh config to ~/.ssh/config:
# For the cluster in Europe
Host gcp_slurm_eu
  HostName 34.91.238.94
  User {your ldap}_sakana_ai
  IdentityFile ~/.ssh/google_compute_engine

# For the cluster in the US	
Host gcp_slurm_us
  HostName 34.136.207.119
  User {your ldap}_sakana_ai
  IdentityFile ~/.ssh/google_compute_engine
Note1: {your_ldap} is your Sakana AI’s email address. (ex. If your email address is fish@sakana.ai, then {your_ldap} is “fish” (namely, User in ssh config should be fish_sakana_ai). If your email address is fish.pez@sakana.ai, {your_ldap} is “fish_pez” (namely, User should be fish_pez_sakana_ai.)
Note2: you don’t need to make an SSH secret key by yourself. The following gcloud compute ssh command will automatically generate  ~/.ssh/google_compute_engine. 

Then, for the first connection, run:
# For EU region
gcloud compute ssh --project=school-of-fish-412803 --zone=europe-west4-c slurm0-login-001


# For US region
gcloud compute ssh --project=school-of-fish-412803 --zone=us-central1-a slurm0us-login-001

For subsequent connections, you can ssh to a login node just by:
# For EU region
ssh gcp_slurm_eu

# For US region
ssh gcp_slurm_us

Please reach out to   if you cannot log in to it.


1.2.2. Check Available Resources on the Login Node
※ From here on, we will explain how to log in to the login node and submit jobs with Slurm, primarily using the Europe region 🇪🇺 as an example. However, the same steps should apply when operating in the US region as well 🇺🇸.

Now you can check if the compute resources are available with sinfo command:

fish_sakana_ai@slurm0-login-001:~$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
a3*          up   infinite      1    mix slurm0-a3nodeset-0
a3*          up   infinite      7   idle slurm0-a3nodeset-[1-7]
debug        up   infinite      4  idle~ slurm0-debugnodeset-[0-3]
(You can forget about debug nodes; These nodes are not equipped with GPUs.)

According to the output above, there are 7 A3 nodes available (8 GPUs and 208 CPU cores each) and someone is using node 0. We are sure that there are enough resources available at this point, but you can check how many GPUs are used if you are interested in:
fish_sakana_ai@slurm0-login-001:~$ sinfo_detail='sinfo --partition=a3 --Node --Format=NodeHost,Gres,GresUsed,CPUsState'
HOSTNAMES           GRES                GRES_USED           CPUS(A/I/O/T)
slurm0-a3nodeset-0  gpu:8               gpu:2               52/156/0/208
slurm0-a3nodeset-1  gpu:8               gpu:0               0/208/0/208
slurm0-a3nodeset-2  gpu:8               gpu:0               0/208/0/208
slurm0-a3nodeset-3  gpu:8               gpu:0               0/208/0/208
slurm0-a3nodeset-4  gpu:8               gpu:0               0/208/0/208
slurm0-a3nodeset-5  gpu:8               gpu:0               0/208/0/208
slurm0-a3nodeset-6  gpu:8               gpu:0               0/208/0/208
slurm0-a3nodeset-7  gpu:8               gpu:0               0/208/0/208
So someone is using 2 GPUs and 52 CPU cores of node 0.
The above is a long command, you can simply use the following to add it to your .bashrc (or .bash_aliases) and use sinfo_detail in the future.
echo "alias sinfo_detail='sinfo --partition=a3 --Node --Format=NodeHost,Gres,GresUsed,CPUsState'" >> $HOME/.bashrc

Now we are ready to submit a training job! There are two ways to submit your job: Interactive job and batch job.
1.2.3. Submit Interactive Jobs
Interactive jobs are closer to the current way to use GPU resources, so it is easier to get started with if you don’t need more than 8 GPUs. Before submitting an interactive job, it is strongly recommended to run a tmux session inside the login node to avoid being disconnected from the compute node.


NOTE
Your Slurm account should be automatically provisioned within approximately 15 minutes after logging in to the login node. If your account has still not been created after 15 minutes, please contact compute-infra@sakana.ai


To submit an interactive job from the login node, you can just run:
srun --pty --partition=a3 --time=01:00:00 --ntasks=1 --gres=gpu:2 bash
It will submit a single task with 1-hour timeout limit, 2 GPUs (and 52 CPUs, 465GB RAM). Then you will directly connect directly to a compute node via SSH, and 2 GPUs and 52 CPUs will be allocated to you:
fish_sakana_ai@slurm0-a3nodeset-0:~$ nvidia-smi
Mon Aug 19 17:38:18 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 80GB HBM3          On  | 00000000:0A:00.0 Off |                    0 |
| N/A   32C    P0              67W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  | 00000000:0B:00.0 Off |                    0 |
| N/A   30C    P0              70W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
fish_sakana_ai@slurm0-a3nodeset-0:~$ nproc
52
Once you log out from the compute node, your resources will be released and others can use GPUs. Please see Slurm Node Tips for environment setup.
1.2.4. Submit Batch Jobs
In case you want to run multi-node training or don’t want to wait for the resource being released, you can submit a job using a batch script. First, you need to create a slurm batch script. For example,
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --time=0:30:00
#SBATCH --partition=a3
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out

# module load
module load cuda/12.1
module load cudnn/8.9.7
module load nccl/cuda-12.1/2.18.3
module load hpcx/2.20


##### multi-node optimization for nccl
UDS_PATH="/run/tcpx-${SLURM_JOB_ID}"

# Only use TCPX for multi-node jobs.
[[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]] && export USE_TCPX=yes || export USE_TCPX=no

# Only use TCPX for multi-node jobs.
if [[ ${USE_TCPX} = "yes" ]]; then
  # Set up NCCL Environment variables
  export NCCL_NET=GPUDirectTCPX_v7
  export NCCL_SOCKET_IFNAME=enp0s12
  export NCCL_GPUDIRECTTCPX_CTRL_DEV=enp0s12
  export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=enp6s0,enp12s0,enp134s0,enp140s0
  export NCCL_CROSS_NIC=0
  export NCCL_ALGO=Ring
  export NCCL_PROTO=Simple
  export NCCL_NSOCKS_PERTHREAD=4
  export NCCL_SOCKET_NTHREADS=1
  export NCCL_MAX_NCHANNELS=12
  export NCCL_MIN_NCHANNELS=12
  export NCCL_DYNAMIC_CHUNK_SIZE=524288
  export NCCL_P2P_NET_CHUNKSIZE=524288
  export NCCL_P2P_PCI_CHUNKSIZE=524288
  export NCCL_P2P_NVL_CHUNKSIZE=1048576
  export NCCL_BUFFSIZE=4194304
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export NCCL_NET_GDR_LEVEL=PIX
  export NCCL_P2P_PXN_LEVEL=0
  export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=${UDS_PATH}
  export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000
  export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
  export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=1000

  export LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH}
else
  unset NCCL_NET
fi
##### multi-node optimization for nccl end


export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

export NUM_GPU_PER_NODE=8
NODE_TYPE="H100"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -bind-to none \
  -x PATH \
  python all_reduce_bench.py
It will spawn 16 MPI processes across 2 nodes (in total 16 GPUs). You can replace the `python` executable with the Python executable in your virtual environment. If you intend to submit a multi-process or multi-threaded job, you can also specify #SBATCH --ntasks-per-node and #SBATCH --cpus-per-tasks  correspondingly to decide the maximum number of cpu cores involved in your job, the former sets the number of instances of your task and the latter sets the number of cpu cores in one task. Normally, the default values work for most GPU use cases.

In case you need to use miniconda, you may need to add
. ~/miniconda3/etc/profile.d/conda.sh
conda activate <YOUR_ENV>
This initializes conda and activates your environment.

Then using this script, you can submit your Slurm batch job as:
sbatch batch.sh

The job status can be checked with the squeue command:
fish_sakana_ai@slurm0-a3nodeset-0:~$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
                24        a3     bash fish_saka R       0:01      1 slurm0-a3nodeset-0
In the above script, the output will be stored in outputs/ folder, so you can monitor the output by, e.g.,

tail -f outputs/test-24.out
Thanks to the tail command and the shared /home directory, the experience should be pretty similar to running the training script directly on the login node.
1.2.5. Manage Submitted Jobs
We can see the status of submitted jobs by:
squeue

It will list the statuses and IDs of submitted jobs. Given a job ID, you can cancel your job by:
scancel <JOB_ID>
1.2.6. GCP Cluster Partitions
🇪🇺 EU Cluster
Partition
Description
a3
A general-purpose GPU cluster. Every user can use this partition.
gpu1
A cluster where you can only submit 1 GPU job; suitable for experimentation with VS Code Remote.
cpu
A general-purpose CPU cluster w/o local SSD.
aisci
A partition for the AI Scientist project.
[⚠️Preemptible] aisci-low
A scavenger partition available to all users. This partition shares the same node sets as the aisci partition. Jobs submitted here can be preempted by AI Scientist Project members, and in that case your job will be canceled and you need to resubmit it on your own!
cpudyn
Automatically spins up CPU nodes when jobs are requested and scales them down once the jobs finish.
cpuide
Jobs cannot be submitted. This CPU node is intended only for Remote SSH access from an IDE.


🇺🇸 US Cluster
Partition
Description
a3
A general-purpose GPU cluster. Every user can use this partition.
aisci
Same as EU.
[⚠️Preemptible] aiscilow
Same as EU. The jobs submitted here can be preempted by AI Scientist Project members, and in that case your job will be canceled and you need to resubmit it on your own!
fugu
A partition for the Fugu project.
[⚠️Preemptible] fugulow
The scavenger partition which is available for all the users. This partition shares the same node sets as the fugu partition. Jobs submitted here can be preempted by Fugu Project members, and in that case your job will be canceled and you need to resubmit it on your own!
cpu
A general-purpose CPU cluster without local SSD.
cpuide
Jobs cannot be submitted. This CPU node is intended only for Remote SSH access from an IDE.


1.2.7. Available Storage Paths
We have the following storage paths available to you:
/home: A workspace shared across GPU and login nodes. We can put code and data here, but we recommend that you avoid storing large amounts of data (more than 6TB) here, since this space is shared by all the researchers.
/mnt/localssd: External NVMe storage attached to each GPU node (i.e. not shared across GPU nodes). This space is to be used for temporary storage, since its contents may be erased during maintenance..

Other storage spaces, e.g. /dev/root, should not be used by the cluster users.


1.3. Connecting to the Cluster Using an IDE (VS Code, Cursor, etc.)
When connecting to the cluster using your IDE (VS Code, Cursor, etc.) via features such as Remote SSH (ref), DO NOT connect directly to the login node.

Instead, add the following entries to your SSH config and connect to gcp_us_ide or gcp_eu_ide for Remote SSH. Please use gcp_us_ide / gcp_eu_ide when using IDE features like Remote SSH.

# -- GCP EU Login Node --
Host gcp_slurm_eu
  HostName 34.91.238.94
  User fish_sakana_ai
  IdentityFile ~/.ssh/google_compute_engine
  ForwardAgent yes
  ServerAliveInterval 60
  ServerAliveCountMax 3
  TCPKeepAlive yes

# Remote SSH for EU
Host gcp_eu_ide
  HostName slurm0-cpuidenodeset-0
  User fish_sakana_ai
  ProxyCommand ssh -W %h:%p gcp_slurm_eu
  ForwardAgent yes
  ServerAliveInterval 60
  ServerAliveCountMax 3
  TCPKeepAlive yes

# -- GCP US Login Node --
Host gcp_slurm_us
  HostName 34.136.207.119
  User fish_sakana_ai
  IdentityFile ~/.ssh/google_compute_engine
  ForwardAgent yes
  ServerAliveInterval 60
  ServerAliveCountMax 3
  TCPKeepAlive yes

# Remote SSH for US
Host gcp_us_ide
  HostName slurm0us-cpuidenodeset-0
  User fish_sakana_ai
  ProxyCommand ssh -W %h:%p gcp_slurm_us
  ForwardAgent yes
  ServerAliveInterval 60
  ServerAliveCountMax 3
  TCPKeepAlive yes


The purpose of this setup is to prevent IDE-related processes from placing excessive load on the login node. 


If you’d like to develop on a compute node with GPUs using VS Code, please follow the instructions in ‘2.2. Development on Slurm Compute Node via VS Code’.

⚠️ Important: Any vscode- or cursor- related processes running on the login node will be automatically killed.

🎩 2. Cluster & Slurm Tips
2.1. Environment Modules
If you want to use cuda-toolkit, cudnn, nccl or hpcx, you need to load it with the module load command. The available modules can be found by module avail command:

fish_sakana_ai@slurm0-login-001:~$ module avail

------------------------------------------------------------------------------------ /opt/apps/modulefiles -------------------------------------------------------------------------------------
   cuda/12.1    cudnn/8.9.7    hpcx/2.20    nccl/cuda-12.1/2.18.3    openmpi/v4.1.x

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
Note1) Please ask  in case you need other versions of the installed modules. (cuda dependencies are bundled with torch if you install it by pip install so that should be enough in most cases.)
Note2) For those who'd like to use zsh etc., you'd need to add "source /etc/profile" in your .zshrc to enable the module command

2.2. Development on Slurm Compute Node via VS Code
It might be preferable to connect VS Code directly to a compute node where you have allocated resources, so that you can test and debug your code without starting an interactive session every time.
Follow the steps below to set up, Step 2 is a one-time setup.
※The following example is for eu region. For the US region, see Tips) Use the settings below for US Clusters.

# Step 1: ssh to the slurm login node 
ssh gcp_slurm_eu

# Step 2: create start_sshd.sh and copy the following to the file

--- Start of start_sshd.sh ---

#!/bin/bash

#SBATCH --job-name="vscode-sshd"
#SBATCH --time=02:00:00
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out

module load cuda/12.1
module load cudnn/8.9.7
module load nccl/cuda-12.1/2.18.3
module load hpcx/2.20

export VSCODE_CUSTOM_PORT=$((10000 + ($SLURM_JOBID % 50000)))

/usr/sbin/sshd -D -p $VSCODE_CUSTOM_PORT -f /dev/null -h $HOME/.ssh/id_rsa -o PubkeyAuthentication=yes -o PasswordAuthentication=no

--- End of start_sshd.sh ---

# Step 3: start sshd
sbatch start_sshd.sh

Now on your laptop, follow the instructions below (remember to fill in your LDAP), and you should be able to SSH to the compute node using slurm-vscode with the desired resources.

# Step 1: Add the following in .ssh/config. This is a one-time setup.

--- Begin of config ---

Host slurm-vscode
  User {your ldap}_sakana_ai
  IdentityFile ~/.ssh/google_compute_engine
  ProxyCommand ssh gcp_slurm_eu "nc \$(squeue --me --name=vscode-sshd --states=R -h -O NodeList:100) \$(squeue --me --name=vscode-sshd --states=R -h -O jobid | awk '{print (\$1 %% 50000) + 10000}')"

--- End of config ---

# Step 2: ssh to the compute node, you can open slurm-vscode in VS Code as well
ssh slurm-vscode

# If you encounter pubkey permission error from the last command, make sure you
# (1) add ~/.ssh/google_compute_engine.pub to .ssh/authorized_keys on the slurm node
# (2) run `ssh-add ~/.ssh/google_compute_engine` on your laptop
Tips) Use the settings below for the US Cluster
For the US Cluster, you need to use the following ssh settings.
# Step 1: Add the following in .ssh/config. This is a one-time setup.

--- Begin of config ---

Host slurm-vscode-us
  User {your ldap}_sakana_ai
  IdentityFile ~/.ssh/google_compute_engine
  ProxyCommand ssh gcp_slurm_us "nc \$(squeue --me --name=vscode-sshd --states=R -h -O NodeList:100) \$(squeue --me --name=vscode-sshd --states=R -h -O jobid | awk '{print (\$1 %% 50000) + 10000}')"

--- End of config ---

# Step 2: ssh to the compute node, you can open slurm-vscode-us in VS Code too
ssh slurm-vscode-us


⚠️FOR WINDOWS USERS⚠️
You need to modify the slurm-VS Code config as follows:

# To run `ssh-add ~/.ssh/google_compute_engine` on your Windows laptop
# (1) Open PowerShell as Administrator
# (2) Set-Service -Name ssh-agent -StartupType Automatic #to set the SSH agent to start automatically.
# (3) Start-Service ssh-agent
# (4) ssh-add C:\Users\<Your User>\.ssh\google_compute_engine


Host slurm-vscode
  User {your ldap}_sakana_ai
  ProxyCommand C:\Windows\System32\OpenSSH\ssh.exe gcp_slurm_eu "nc $(squeue --me --name=vscode-sshd --states=R -h -O NodeList:100) $(squeue --me --name=vscode-sshd --states=R -h -O jobid | awk '{print ($1 %% 50000) + 10000}')"
2.3. Installing and Using Specific Packages
Note 
[Jan 16, 2026]
We highly recommend using uv for Python package management instead of other management tools like conda, pyenv, etc.

2.3.1. conda
If you want to install conda inside your home folder, please follow these steps.

download + install miniconda locally
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
prepend local conda path to PATH
export PATH="/home/$USER/miniconda3/condabin:$PATH"
open .bashrc and remove (if present) everything for default conda initialize:
# >>> conda initialize >>>
...
# <<< conda initialize <<<
in a new shell, re-init the local conda installation
~/miniconda3/condabin/conda init
create a .condarc, then add the following:
y
add the following to the end of the .condarc file (This removes the global default pkgs_dirs/envs_dirs from the conda configurations):
(Please replace {user_here} with your Slurm cluster username (e.g. fish_sakana_ai))
pkgs_dirs:
  - /home/{user_here}/.conda/pkgs
envs_dirs:
  - /home/{user_here}/.conda/envs
 
(optionally, if you do not want to activate the base conda env by default, run the following)

conda config --set auto_activate_base false

2.3.2. pyenv
curl https://pyenv.run | bash

# add to .bashrc
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pyenv install 3.xx.xx
pyenv global 3.xx.xx
python --version

2.3.3. Go
First, go to the Go website and download the archive goX.XX.XX.linux-amd64.tar.gz (where X.XX.XX is the version number of Go you want to install).
Then extract the content, move it somewhere under your home directory (say ~/bin) by:

tar xf  goX.XX.XX.linux-amd64.tar.gz
mkdir ~/bin
mv go ~/bin/
Then you need to add the following line to your .bashrc:

export PATH="$PATH:$HOME/bin/go/bin"
2.3.4.  Git LFS
You can download Git LFS binary (“Linux AMD64” ones) with any version from the release page. Then you can extract its content and put the “git-lfs” binary in (say) ~/bin folder.
Please do not run install.sh, just put git-lfs to some folder inside your home directory.

Then add th binary directory to your PATH:
export PATH="$PATH:$HOME/bin"
Then you need to run
git lfs install
2.3.5. ffmpeg
There is a static build of ffmpeg where we can download and just run the binary out of the box.
Please follow the instruction here and put the binary somewhere inside your home directory: https://www.johnvansickle.com/ffmpeg/faq/ 




2.4. Add GitHub ssh key
You can add your GitHub ssh key by running the following command on your local machine:

ssh-add <path_to_github_private_key>

Then in your local ~/.ssh/config add the ForwardAgent yes to the gcp_slurm_eu and slurm_vscode hosts. If you encounter issues with agent forwarding in VS Code, this issue is helpful. In particular, running:

export SSH_AUTH_SOCK=$(ls -t /tmp/ssh-**/* | head -1)

2.5. Docker on GCP Clusters
2.5.1.  🐳 How to Use Docker on Slurm
If you use docker on Slurm, please specify --gpus option in docker run as:
--gpus="\"device=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}\""
Otherwise, your process uses GPUs that are not assigned to you!

See https://github.com/NVIDIA/nvidia-container-toolkit/issues/211#issuecomment-1903927569 

Also, please stop the Docker container explicitly before exiting your job:
docker stop <container_name>
docker rm <container_name>
Otherwise your docker process may not be killed by slurm.
2.5.2.  🐳 Rootless docker on GPU nodes
To avoid pulling heavy images to the boot disk with rootful docker, you can leverage rootless docker to pull images to local SSD instead.

You can use rootless docker with the following steps (Please run those setup commands on Login Node):

Install rootless docker by:
mkdir -p ~/.config/docker
dockerd-rootless-setuptool.sh install --force
Add the following lines at the end of your ~/.bashrc:
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
Modify the ExecStart in ~/.config/systemd/user/docker.service as:
ExecStart=/usr/bin/dockerd-rootless.sh --data-root /mnt/localssd/<YOUR_USER_NAME>/rootless_docker
where --data-root /mnt/localssd/<YOUR_USER_NAME>/rootless_docker argument was added. <YOUR_USER_NAME> can be checked with the echo $USER command.

Then you need to run:
echo '{"data-root": "/mnt/localssd/'"$(whoami)"'/rootless_docker"}' > ~/.config/docker/daemon.json

systemctl --user stop docker
systemctl --user daemon-reload

This reloads the Docker configuration.

To run docker command, you first need to enable lingering on each GPU node in your sbatch script. For example,

#!/bin/bash

#SBATCH --job-name="vscode-sshd"
#SBATCH --time=1-0:00:00
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --nodelist=slurm0us-a3nodeset-2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out

module load cuda/12.1
module load cudnn/8.9.7
module load nccl/cuda-12.1/2.18.3
module load hpcx/2.20

export VSCODE_CUSTOM_PORT=$((10000 + ($SLURM_JOBID % 50000)))

# turn on lingering
if ! loginctl show-user $(whoami) | grep -q "Linger=yes"; then
    echo "Linger is disabled. Enabling now..."
    loginctl enable-linger $(whoami)
fi

# Wait until lingering is fully turned on (max 30 seconds)
for i in {1..30}; do
    if [ -d "/run/user/$(id -u)" ]; then
        echo "/run/user/$(id -u) is now available!"
        break
    fi
    echo "Waiting for /run/user/$(id -u)... ($i/30)"
    sleep 1
done


# Launch dockerd-rootless.sh in the background

dockerd-rootless.sh &
# wait for Dockerd daemon to start (max 30 seconds)
for i in {1..30}; do
    if docker info > /dev/null 2>&1; then
        echo "Docker is up and running!"
        break
    fi
    echo "Waiting for Docker to start..."
    sleep 1
done

# handle errors in case dockerd didn't start correctly
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker did not start correctly!"
    exit 1
fi

docker run hello-world

You can change the docker run hello-world part to your own command. 


In case of an interactive job, you can simply run:
loginctl enable-linger $(whoami)
dockerd-rootless.sh &
docker run hello-world

2.6.  🤝 How to share data across the two clusters

 [IMPORTANT] Notes
Cross-Region Data Sharing: Cross-region data sharing typically involves significant time and cost overhead, so it is generally not recommended unless absolutely necessary.
Data Transfer Costs: Cloud Storage incurs costs for data storage and transfer, especially for inter-region traffic. Verify cost estimates beforehand.
See: Pricing  |  Cloud Storage  |  Google Cloud 

⚫️ How to share small amounts of data using rsync and scp
You can run scp across GCP machines. To do that, from your local machine,
scp -r gcp_slurm_eu:/path/to/source gcp_slurm_us:/path/to/destination
Since scp is generally slow, if the data size is large, please use the data transfer method through GCS bucket as is detailed below.
⚫️ How to share large amount of data through a GCS bucket
Here is how to share data between clusters in the EU and US regions. Below, we describe how the user fish_sakana_ai can transfer data from /home/fish_sakana_ai/data on the EU cluster to /home/fish_sakana_ai/data on the US cluster.
1. Prerequisites
SSH access is available for both the EU node (gcp_slurm_eu) and the US node (gcp_slurm_us).
The gsutil tool is installed and accessible on both nodes.
Cloud Storage can be used as an intermediary for temporary data storage and transfer.
2. Overview of Steps
Upload data from the EU node to a Cloud Storage bucket.
Download the data from the Cloud Storage bucket to the US node.
3. Detailed Steps
Step 1: Create a Cloud Storage Bucket
From the US node or your local environment, create a bucket. For example:
gcloud auth login
gsutil mb -l US gs://{your_name}_gsu
The -l US flag ensures that the bucket is created in the US region.
Replace {your_name} with a prefix like your username (e.g., fish).
Step 2: Upload Data from the EU Node to the Cloud Storage Bucket
Log in to the EU node:
ssh gcp_slurm_eu
Upload the contents of /home/fish_sakana_ai/data to the Cloud Storage bucket:
gsutil -m rsync -r /home/fish_sakana_ai/data gs://{your_name}_eu-to-us-transfer-bucket
-m: Enables parallel processing to improve transfer speed.
-r: Recursively syncs all directories and files.
Step 3: Download Data from the Cloud Storage Bucket to the US Node
Log in to the US node:
ssh gcp_slurm_us
Download the data from the Cloud Storage bucket to /home/fish_sakana_ai/data:
gsutil -m rsync -r gs://{your_name}_eu-to-us-transfer-bucket /home/fish_sakana_ai/data
By following these steps, you can efficiently transfer the /home/fish_sakana_ai/data directory from the EU region's login node to the US region's login node.
Ref:
Syncing from GCP [From GCP QA mailing list]
GCP fast sync guide



2.7 👬 Shared Directory Under /home
On the GCP EU/US clusters, there is a shared directory located at /home/shared.
If you need to share data with other users on the same cluster, please use this directory.

When creating a new directory under /home/shared, make sure to name it after your own username. Directories with any other names may be deleted!!

# ex)
cd /home/shared
mkdir $USER # Directory names other than your username are not allowed.

Tips
If you want to share data that originally resides under your own /home directory, please make active use of symbolic links under /home/shared.




2.8. [IMPORTANT] Data Policy for Departed Members
When a member leaves the company, their /home/{RETIRED_USER} directory is automatically moved to /home/backup/{RETIRED_USER}.
The migrated data will generally be deleted automatically after approximately one week. When a team member is leaving, please make sure to take any necessary actions in advance, such as uploading required data to GCS or changing directory permissions so that the necessary files can be moved.


🔥Troubleshooting
Notice
If you can’t resolve your issues on GCP, join the cluster_users@sakana.ai group and post your question there. Once the problem is fixed, please add both the issue and its solution to this “Troubleshooting” section.

Q) How to install additional packages on all nodes?
Please make a PR for the following script. Once it’s approved, it’ll be installed on all nodes in both EU and US clusters. 
https://github.com/SakanaAI/gcp-slurm-admin/blob/main/ansible/setup-packages.yml 
Q) Nodes are in the DRAIN state and we can’t submit jobs
If you encounter the node with “DRAIN” state, the node is temporarily down due to the anomaly detection system of slurm. Please contact  to get it back to the normal state.
Q) When I submit a job, my username is shown with some number in the `squeue` list rather than my name. Also `whoami` raised an error.
In this case, please check whether you modified LD_LIBRARY_PATH in your ~/.bashrc and if so, append only necessary custom paths to it. Otherwise it may override some libraries slurm is using.
Q) My username is not recognized inside my slurm job inside compute nodes.
The same as above.
Q) There is some issue when using screen, PATH and LD_LIBRARY_PATH are not properly set
We recommend tmux over screen; Those two terminal multiplexers work in a very different way, and the official recommendation from environment modules developers is tmux.
Q) When we spawn multiple tmux session on the same compute node from different interactive jobs, all of the sessions tries to access one gpu
This is caused by the shared tmux socket. We can mitigate this issue by adding the following snippet to ~/.bashrc:

tmux() {
        command tmux -L "${SLURM_JOBID:-default}" "$@"
}
So that the different slurm job connects to different tmux sockets.
Q) Connection Error: Tried to launch distributed communication on port ‘29500’...
In this case, you need to modify the master port of your GPU communication. This can be done, e.g. for accelerate by
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))
accelerate launch --main_process_port $MASTER_PORT <SCRIPT>
To avoid the port collision with other processes.
Q) Receiving the following error when trying to connect to slurm-vscode

nc: port number invalid: slurm0-a3nodeset-4
Connection closed by UNKNOWN port 65535

This error comes from having multiple vscode-sshd jobs spawned. Your current jobs can be viewed with squeue --user $USER and can be canceled with scancel <job_id>.
Q) Cannot connect to the NFS server after running sbatch scripts including docker compose
When running multiple jobs using the same Docker Compose setup simultaneously on a single server, conflicts with subnets and network interfaces may occur. This can lead to issues in the local network, such as being unable to connect to the NFS server. (Docker Compose automatically generates a network named “project_name_default,” and containers join that network.)
To avoid this issue, you can use docker run and configure it to utilize the default bridge network (docker0), which prevents network conflicts.
For more details about problems encountered in the past, please refer to the following documentation.
Ref: CPU node malfunction investigation
Q) Cannot run multiple docker containers at the same time
Try executing the following commands and increasing the usage limit of inotify:
echo {appropriate_num} | sudo tee /proc/sys/fs/inotify/max_user_instances
echo {appropriate_num} | sudo tee /proc/sys/fs/inotify/max_user_watches
Q) Running multiple docker make the node I am using unreachable. What’s happening?
When creating a new docker a network is created, this network use a range of ip adress define by default here. In some cases, the ip used by the new created docker network can collide with the NFS ip adress, leading to the node being unreachable.
To check if this is happening you can first you can check your NFS ip adress by running the following:

findmnt -t nfs,nfs4 -o TARGET,SOURCE,FSTYPE,OPTIONS
If your adress is in the 192.168.0.0 you have a risk to collide. To fix it you can change your config file in rootless mode this should be ~/.config/docker/daemon.json. 
One simple fix is to remove the conflicting ip and setting the default-adress setting daemon.json to:

{
   "runtimes": {
       "nvidia": {
           "path": "nvidia-container-runtime",
           "runtimeArgs": []
       }
   },
   "default-address-pools": [
       {"base":"172.17.0.0/16","size":16},
       {"base":"172.18.0.0/16","size":16},
       {"base":"172.19.0.0/16","size":16},
       {"base":"172.20.0.0/14","size":16},
       {"base":"172.24.0.0/14","size":16},
       {"base":"172.28.0.0/14","size":16}
   ]
}
 You may also want to update the size from 16 to 24. This parameter control the range of the ips increasing will be useful if you run a lot of dockers. The above enable the creation of 15 different networks. Passing the size to 24 raise the number of possible network to 3840.

Q) How to see total GPUs allocated / user
Add the following alias in .bashrc:
alias sinfo_user='squeue -O "UserName:30,tres-per-node:16,tres-per-job:16,State:10" | awk '\''$4=="PENDING"{next} !($2~/gpu:/||$3~/gpu:/){next} {cnt=0; if(match($2,/gpu:([1-9][0-9]*)/,a))cnt=a[1]; else if(match($3,/gpu:([1-9][0-9]*)/,b))cnt=b[1]; if(cnt>0)usr_gpu[$1]+=cnt} END{for(u in usr_gpu)print u,usr_gpu[u]}'\'' | sort -k2 -nr | awk '\''{ print $0; sum += $2 } END { print "-----\nTOTAL:", sum }'\'''

Then, reload .bashrc and run sinfo_user:
$ sinfo_user
fuga_sakana_ai 16
fish_sakana_ai 14
hoge_sakana_ai 10
...
-----
TOTAL: 102

Q) Cannot execute sudo on a node
Execute the following command and try sshing again:
# US
gcloud compute ssh --project=school-of-fish-412803 --zone=us-central1-a slurm0us-a3nodeset-xx --command="exit"

# EU
gcloud compute ssh --project=school-of-fish-412803 --zone=europe-west4-c slurm0us-a3nodeset-xx --command="exit"

Q) flash_attn, GLIBC ImportError
When using flash_attn package, using the latest version will likely throw the following error:
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found

A workaround is to downgrade to previous version, for instance (see this thread (Flash-Attention GitHub Repository) for more information):
pip install flash-attn==2.7.2.post1 --no-build-isolation

Q) Port forwarding
GCP US:
Add the contents of your local ~/.ssh/google_compute_engine.pub to ~/.ssh/authorized_keys on the login node. This will propagate to the compute nodes.
Add this to your local ~/.ssh/config
Host slurm0us-a3nodeset-? slurm0us-a3nodeset-??
  HostName %h
  User {your_ldap}_sakana_ai
  ProxyJump gcp_slurm_us
  LocalForward 5000 localhost:5000
Finally, ssh from local to the compute node.

Q) How to set up a shared data repository on Slurm nodes
💡 Issue
Home directories (/home/user_xxx) are not world-readable, so it’s not possible to share experiment outputs directly under them.
Uploading/downloading via Hugging Face or GitHub is inefficient when data must remain local for analysis.
A shared repository is needed for collaborative projects on the cluster.
Two main approaches are recommended:
Use Google Cloud Storage (GCS) buckets mounted via gcsfuse


Allows multiple users to read and write shared data.
Accessible from multiple clusters, making it useful during high-load periods.
Slightly slower than local disk, but convenient for centralized access.
Official setup guide
  Mounting buckets with gcsfuse


Use a shared local directory with group-based permissions
A shared Unix group can be created for the project.
The directory can be made accessible only to group members:
mkdir /shared/project_data
chown <owner>:<project_group> /shared/project_data
chmod 770 /shared/project_data
Avoid placing shared data under home directories, since top-level permissions block access.
Ask  to make the group

Q) How to use CUPTI?
Please extend your LD_LIBRARY_PATH with $CUDA_HOME/extras/CUPTI/lib64.

Q) Cannot install codex on the GCP nodes!! 😢
You might be trying to install it globally with npm. Regular users don't have sudo, so normally you can't install that way. If you install node using n (https://github.com/tj/n) or nvm (https://github.com/nvm-sh/nvm) and then install with npm in your user space, you should be able to install it.

