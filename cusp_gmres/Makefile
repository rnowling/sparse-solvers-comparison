CC=nvcc
PROGRAM=cusp_gmres

$(PROGRAM): $(PROGRAM).cu
	$(CC) -o $(PROGRAM) $(PROGRAM).cu

.PHONY: clean

clean:
	rm -f $(PROGRAM)
