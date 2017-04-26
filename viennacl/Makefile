CC=nvcc
PROGRAM=viennacl

$(PROGRAM): $(PROGRAM).cu
	$(CC) -o $(PROGRAM) -L/usr/local/include $(PROGRAM).cu

.PHONY: clean

clean:
	rm -f $(PROGRAM)
