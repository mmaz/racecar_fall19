# Server Usage

!!! danger "Keep this in mind!"
     The servers are a shared resource! Coordinate the times you are using the server with the other teams so that you do not overlap. And **remember to shut down your Jupyter Notebook or training script** after you are done. Do not leave a training script running - it will reserve all available GPU memory, and prevent other teams from training.

     Please ask the instructors before running any commands with `sudo` rights.

     Lastly, please do **not** reboot the server! Again, ask if you have questions.

## Installing Dependencies

There's nothing to do! The dependencies you need for the **Imitation learning lab** are already installed in the `imitation_learning` conda environment. 

When you connect, run `conda activate imitation_learning`).

## Connecting to the server over SSH

At the terminal, the following command will SSH into the server, enable x-forwarding, and also forward a port for Jupyter Notebook access:

!!! note
    You will need to replace `$USERNAME` and `$IP_ADDRESS` below with the appropriate values (ask the instructors)

```shell
ssh -L 8888:localhost:8888 -X  $USERNAME@$IP_ADDRESS
```

Once you are connected, run `ls` - you have an empty directory for your team already created in the home directory:

```shell
$ ls
[snip] team14 team23 team37 team52 team56 team65 [snip] 
```

After you are SSH-ed in, remember to `cd` into your team's directory and activate the `imitation_learning` environment.

### Adding a shortcut for easy SSH access:

**This step is optional**

You can simplify the process of connecting to the server further by adding the following information to your **local** SSH config file:

!!! danger "Warning"
    Make sure you are editing the config file __on your personal laptop__ (not the server!) before proceeding:

    `$ vi ~/.ssh/config`

!!! note
    You will need to replace `$USERNAME` and `$IP_ADDRESS` below with the appropriate values (ask the instructors in the Slack chatroom)

```
Host 6a01
    User $USERNAME
    HostName $IP_ADDRESS
    ForwardX11 yes
    IdentityFile ~/.ssh/id_rsa
    LocalForward 8888 127.0.0.1:8888
    ControlPath ~/.ssh/controlmasters_%r@%h:%p
    ControlMaster auto
```

This sets up port-forwarding (for Jupyter), [SSH multiplexing](https://en.wikibooks.org/wiki/OpenSSH/Cookbook/Multiplexing#Setting_Up_Multiplexing), and X-forwarding once and for all.

Afterwards you can SSH into the server just by typing:

```shell
$ ssh 6a01
```

at your local command prompt.

### Convenient SCP

With the shortcut documented above, you can also SCP directories and files to the server easily, e.g., to copy to your team's directory:

```shell
$ scp -r local_directory_w_training_images 6a01:~/teamN/ # where N is your team ID
```

### (Faster) SCP Copying:

`scp` (secure copy) is a tool used for copying files to or from a remote location over a network. It follows the same format as `cp`:

`scp -r $SOURCE_LOCATION $DESTINATION_LOCATION`

where `-r` specifies *recursive* usage (i.e., the ability to copy directories and subdirectories).

However, copying each `.jpg` training image from a source to a destination one-by-one is slow. You might instead want to create a single `.zip` or `.tar.gz` file of your directory. For instance, if you are on your local machine, and you want to zip up a folder called `data_12_02_2019_8-34/` to copy over to a training laptop:

```bash
# format: tar -czvf $DESIRED_FILENAME $DIRECTORY
$ tar -czvf data_12_02_2019_8-34.tar.gz  data_12_02_2019_8-34/
```

Then, to `scp` the file over, if you are NOT using the config shortcut documented above: 

```bash
# replace USERNAME and IPADDRESS
$ scp data_12_02_2019_8-34.tar.gz $USERNAME@IPADDRESS:~/teamN
```
If you are using the shortcut you can instead run the following command:

```bash
$ scp data_12_02_2019_8-34.tar.gz 6a01:~/teamN
```

Note that the tilde `~` after the `:` specifies a destination path (teamID) relative to `$USERNAME`'s home directory.

The folder can be expanded on the remote laptop (after ssh-ing in) via:

```bash
$ cd ~/teamN
$ tar -xzvf data_12_02_2019_8-34.tar.gz 
```

### No-password logins

If you would like to avoid being asked for a password each time, you can generate a local identity file:

<https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>

Then, use the following command to connect to the server from your laptop.

```shell
# only required once
$ ssh-copy-id 6a01
```

which will copy your public key to the server.

## Opening Jupyter Notebooks over SSH

!!! note
     If you have Jupyter Notebook running locally, shut it down first (or restart it and change the port to something besides the default port `8888`, so that the local port used by Jupyter does not conflict with the SSH-forwarded one)

     **On your local computer:**

     `jupyter notebook --port 8889`
    
     for example.

Once you have SSHed into the computer, you can start jupyter in your team's folder (include `--no-browser` so that Firefox does not try to load over SSH - which will be annoyingly slow)

```shell
$ cd teamN/
$ conda activate imitation_learning
$ jupyter notebook --no-browser
```

Jupyter will provide a URL for you to use in your local computer's browser. Copy and paste it into your browser, e.g.,

```
Or copy and paste one of these URLs:
   http://localhost:8888/?token=r4nd0mh3xstring
```

!!! note
    It is highly recommended to run a program like `tmux` or `screen` after first logging in, so that your work is not lost in case your SSH connection is interrupted. `tmux` is already installed on the server. 

    * [Here](https://danielmiessler.com/study/tmux/) is a simple guide that introduces `tmux`. 
    * Also, [here](https://gist.github.com/andreyvit/2921703) is a cheatsheet for quick reference.

```shell
$ cd teamN/
$ tmux
$ conda activate imitation_learning
$ jupyter notebook --no-browser
```

!!! danger "Advanced usage"
     **Concurrent usage by multiple teams:** If you and another team agree beforehand, you can [limit the memory TensorFlow allocates while training](https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth), and both teams can train on the server simultaneously. Again remember to shut down jupyter and/or your python training scripts when you are done. Additionally, note that if both teams want to concurrently run Jupyter notebook instances, you cannot share the same port (by default, `8888`). Different ports will be auto-assigned. You can manually specify a port with `jupyter notebook --no-browser --port ####`, for example, `--port 8765`). You will need to ensure the port for your notebook instance is forwarded via ssh.
