Tmax=20;
kh = 0.8;
ki = 0.8;
kp = 0.1;

A = [-kh,   0,   0, 0;
      kh, -ki,   0, 0;
      0,   ki, -kp, 0;
      0,    0,  kp, 0];

z0 = [1, 0, 0, 0]';

ts = 0.2;
maxT = 7;


nT = 7 * 24 / 2;
allTs = 1:nT;

zs = zeros(nT, 4);
zs(1, :) = z0;

for t = 2:nT
    zs(t, :) = zs(t-1, :)' + ts * A * zs(t-1, :)';
end

set(groot,'defaultLineLineWidth',2.0)
plot(allTs / 12, zs);
legend(['H'; 'I'; 'P'; 'M'])