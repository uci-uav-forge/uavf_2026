from pathlib import Path
import subprocess
from time import sleep

# clear score files
all_score_files = list(Path(".").glob("mission*.txt"))
for f in all_score_files:
    f.unlink()

# run sim 10 times
CONTINUOUS = True
if CONTINUOUS:
    subprocess.run("./setup_sim.sh", shell=True, text=True)
sleep(30)
for run_num in range(10):
    if not CONTINUOUS:
        try:
            subprocess.run(
                "SKIP_WAYPOINTS=1 ./setup_and_run_mission.sh",
                shell=True,
                text=True,
                check=True,
                timeout=5 * 60,
            )
        except subprocess.TimeoutExpired:
            print(f"Run {run_num} timed out")
        subprocess.run("pkill -9 -f setup_and_run_mission", shell=True, text=True)
    else:
        sleep(10)
        try:
            subprocess.run(
                "SKIP_WAYPOINTS=1 ./run_mission.sh",
                shell=True,
                text=True,
                check=True,
                timeout=3 * 60 if run_num > 0 else 5 * 60,
            )
        except subprocess.TimeoutExpired:
            print(f"Run {run_num} timed out")
        except subprocess.CalledProcessError:
            print(f"Run {run_num} CalledProcessError")
        subprocess.run("pkill -9 -f run_mission.sh", shell=True, text=True)

subprocess.run("./cleanup.sh", shell=True, text=True)

# calculate average score from files
all_score_files = list(Path(".").glob("mission*.txt"))

total_score = 0
for score_file_path in all_score_files:
    with open(score_file_path, "r") as f:
        total_score += float(f.read().strip())

print(f"Average score: {total_score/len(all_score_files)}")
