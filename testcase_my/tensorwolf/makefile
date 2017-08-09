OBJS = kernel.o
CFLAGS = -O4 -fPIC
kernel.so: kernel.o
	g++ ${CFLAGS} -shared kernel.o -o kernel.so
	rm kernel.o
kernel.o: kernel.cpp
	g++ ${CFLAGS} kernel.cpp -c -o kernel.o
clean:
	rm kernel.so
	rm kernel.o
