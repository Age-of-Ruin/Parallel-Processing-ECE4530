compile_serial:
	mpicxx -o heat HeatEquation_serial.cpp -I/usr/local/include -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
compile_parallel:
	mpicxx -o heat HeatEquation_parallel.cpp -I/usr/local/include -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
clean:
	rm *jpg
	