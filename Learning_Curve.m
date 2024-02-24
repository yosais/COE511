% Generate 1000 random integer rewards

rng(0,'twister');

size = 1000;
reward = randi([0 10],1,size);

mov_avg = zeros(1, size);

for i = 1:size
    mov_avg(i) = sum( reward(1:i) ) / i;
end

plot(mov_avg);
xlabel('Time Steps');
ylabel('Moving Average');

yline(mov_avg(size), '--r', 'LineWidth',1.25);

