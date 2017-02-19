CC=nvcc
PROGRAM=cusp_cg

$(PROGRAM): $(PROGRAM).cu
	$(CC) -o $(PROGRAM) $(PROGRAM).cu

.PHONY: clean

clean:
	rm -f $(PROGRAM)
