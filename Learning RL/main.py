import subprocess

# List of your python scripts
scripts = ["MazegamewithA2C.py", "MazegamewithDQN.py", "MazegamewithPPO.py"]

# Create a list to hold the processes
processes = []

for script in scripts:
    # Start each script in a separate process
    process = subprocess.Popen(["python", script])
    processes.append(process)

# Wait for all processes to finish
for process in processes:
    process.wait()
