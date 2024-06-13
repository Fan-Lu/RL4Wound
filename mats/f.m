function dy = f(t, y)

global N;
global kh;
global ki;
global kp;

y=reshape(y, 1, 4);
dy=zeros(1, 4);

dy(1, 1) = -y(1,1)*kh;                          
dy(1, 2) = y(1,1)*kh - y(1,2)*ki;
dy(1, 3) = y(1,2)*ki - y(1,3)*kp;
dy(1, 4) = y(1,3)*kp;


y=reshape(y, 4, 1);
dy=reshape(dy, 4, 1);

end