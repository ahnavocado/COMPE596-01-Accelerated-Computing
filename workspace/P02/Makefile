CC = gcc
CFLAGS = -Wall -fopenmp
EXEC_SERIAL = node_single
EXEC_PARALLEL = node

all: node_single node

node_single: Node_single.c Node.h
	$(CC) $(CFLAGS) Node_single.c -o $(EXEC_SERIAL)

node: Node.c Node.h
	$(CC) $(CFLAGS) Node.c -o $(EXEC_PARALLEL)

# Run serial version with fixed OMP_NUM_THREADS
run_s1: node_single
	(unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1; ./$(EXEC_SERIAL))

run_s2: node_single
	(unset OMP_NUM_THREADS; export OMP_NUM_THREADS=2; ./$(EXEC_SERIAL))

# Run parallel version with fixed OMP_NUM_THREADS and arguments
run1: node
	(unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1; ./$(EXEC_PARALLEL) 0)

run2: node
	(unset OMP_NUM_THREADS; export OMP_NUM_THREADS=2; ./$(EXEC_PARALLEL) 1)

run4: node
	(unset OMP_NUM_THREADS; export OMP_NUM_THREADS=4; ./$(EXEC_PARALLEL) 2)

run16: node
	(unset OMP_NUM_THREADS; export OMP_NUM_THREADS=16; ./$(EXEC_PARALLEL) 4)

# Run parallel version with multiple thread counts
run_parallel:
	@echo "Running parallel version with different thread counts..."
	@for T in 4 8 16 32 64 128 256 512 1024; do \
		echo "Running with $$T threads..."; \
		unset OMP_NUM_THREADS; export OMP_NUM_THREADS=$$T; ./$(EXEC_PARALLEL); \
		echo ""; \
	done

clean:
	rm -f $(EXEC_SERIAL) $(EXEC_PARALLEL)

