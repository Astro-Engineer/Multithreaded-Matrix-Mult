NVCC = nvcc
NVCCFLAGS = -O3

all: p1 p2

p1: p1.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

p2: p2.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f p1 p2
