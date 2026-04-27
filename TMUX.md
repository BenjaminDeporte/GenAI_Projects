https://gist.github.com/MohamedAlaa/2961058

tmux shortcuts & cheatsheet

start new:tmux
start new with session name:tmux new -s myname
attach: tmux a  #  (or at, or attach)
attach to named: tmux a -t myname
list sessions: tmux ls
kill session: tmux kill-session -t myname
Kill all the tmux sessions: tmux ls | grep : | cut -d. -f1 | awk '{print substr($1, 0, length($1)-1)}' | xargs kill

In tmux, hit the prefix ctrl+b (my modified prefix is ctrl+a) and then:

List all shortcuts
to see all the shortcuts keys in tmux simply use the bind-key ? in my case that would be CTRL-B ?

Sessions
:new<CR>  new session
s  list sessions
$  name session


### survival kit

Minimal tmux sequence to run it:

1. Start a named session and run the script directly
tmux new-session -s sft-medical 'cd src && python SFT_medical_reasoning.py'

2. Detach while it runs
Press Ctrl+b, then d

3. Re-attach later
tmux attach -t sft-medical

4. End session when done
tmux kill-session -t sft-medical

If you want the interactive version instead:

tmux new -s sft-medical
cd src
python SFT_medical_reasoning.py