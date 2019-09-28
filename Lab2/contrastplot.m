x = 0:1:255; %input pixel value
y = 0:1:255; %output pixel value

minval = 20;
maxval = 40;

y(1:minval) = 0;
y(maxval:end) = 255;   

%Plot
figure(1)
plot(x,y)
xlabel('Input Pixel Values')
ylabel('Output Pixel Values')
title('Graph of Contrast Enhance Function')