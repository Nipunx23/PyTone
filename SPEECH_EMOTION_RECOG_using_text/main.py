import time
start_time = time.time()

# Execute the code.
for i in range(1000000):
    pass

# Stop the timer.
end_time = time.time()

# Calculate the number of iterations per second.
iterations_per_second = 1000000 / (end_time - start_time)

print(start_time,end_time,iterations_per_second)