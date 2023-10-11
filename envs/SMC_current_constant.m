clc
clear all
close all

global voltage_output
global t

t = tcpclient("192.168.4.1",9997); % IP connection to Raspberry PI




delay = 0.5;   % change based on sampling time of the microscope
alp = 1;
a = 1;
%waitMax = 20;
jjj = 1;







Nnc = 1800;    % Number of points

timeref = 1:Nnc+1;



% desired value generation +deravative generation if it is time varying
piii = -5*pi:0.025:5*pi;
piii2 = ceil(sin(piii));
piii3 = ceil(cos(piii));

piii22 = sin(piii);
piii33 = cos(piii);

xx = 1;




% plot(10*ones(1,Nnc+1)+piii22(1:Nnc+1),'b','linewidth',1.5);
% hold on
% plot(10*ones(1,Nnc+1)+piii33(1:Nnc+1),'r','linewidth',1.5);
% hold off

r(timeref)=400*ones(1,Nnc+1);     % constant signal









% amp_coef = 1.5;
% bias_coef = 61;
% 
% r(timeref) = bias_coef*ones(1,Nnc+1)+amp_coef*piii22(1:Nnc+1);
% dot_r(timeref) = bias_coef*ones(1,Nnc+1)+amp_coef*piii33(1:Nnc+1);

% r11(timeref)=10*ones(1,Nnc+1)+0.0125*(timeref).*ones(1,Nnc+1);
% r22(timeref)=15*ones(1,Nnc+1)-0.0125*(timeref).*ones(1,Nnc+1);
% r(1:Nnc/2)=r22(1:Nnc/2);
% r(Nnc/2+1:Nnc+1)=r11(1:Nnc/2+1);
%
% r111(timeref)=10*ones(1,Nnc+1)+0.0125*(timeref).*ones(1,Nnc+1);
% r222(timeref)=15*ones(1,Nnc+1)-0.0125*(timeref).*ones(1,Nnc+1);
%
% r2(1:Nnc/2)=r111(1:Nnc/2);
% r2(Nnc/2+1:Nnc+1)=r222(1:Nnc/2+1);

% r(timeref)=1.5*ones(1,Nnc+1)+xx*piii2(1:Nnc+1);%-0.01*(timeref).*ones(1,Nnc+1);
% plot(sin(piii),'b','linewidth',1.5);

%r2(timeref)=1.5*ones(1,Nnc+1)+xx*piii3(1:Nnc+1);%-0.01*(timeref).*ones(1,Nnc+1);

% r1(timeref)=8*ones(1,Nnc+1);
%r(timeref)=3.25-3*square_t((2*pi/150)*timeref); % A square wave input
%r(timeref) = 40.1-20*square_t((2*pi/45)*timeref); % A square wave input
%r1(timeref) = 4.1-4*square_t1((2*pi/150)*timeref); % A square wave input
% plot(r(1:end),'b','linewidth',1.5)
% hold on
% plot(r2(1:end),'r','linewidth',1.5)
%rr(timeref) = 4.1-4*square_t((2*pi/150)*timeref); % A square wave input
%r(timeref)=3.25*ones(1,Nnc+1)-3*(2*rand(1,Nnc+1)-ones(1,Nnc+1)); % A noise input
ref = r(1:Nnc);  % Then, use this one for plotting
%ref2 = r2(1:Nnc);
%ref1 = r1(1:Nnc);  % Then, use this one for plotting

time = 1:Nnc;
%time = T*time; % Next, make the vector real time

%load('test_data_read_xlsx_v2.mat');
% in = [pulse_f;t];
in = ref;
%in2 = ref2;

len1 = length(in);
%runs = 1;

x_d = in; % desired trajectory
d(1) = x_d(1);
%x = [];
x = zeros(1,1); % initial actual output
%x = temp_current(active_electrode1,i1);
dot_x = zeros(1,len1); % if input trajectory is constant

e(1) = 0; % initial error

u(1) = 0; % initial control input
nu(1) = 0; % initial artificial control input

if e>0
    K_pos = 0.4; % tunable positive parameter 1
    ro_gain = .08; % tunable positive parameter: .2
else
    K_pos = 0.4; % tunable positive parameter 1
    ro_gain = .008; % tunable positive parameter: .05
end

A_max = 1; % maximum Voltage to be applied
min_val=0; % minimum Voltage to be applied

T_samp = 1; % sampling time for the denominator



%Do not change this line
voltage_output = ["F","F","F","F","F","F","F","F","F","F","F","F","F","F","F","F"];
%voltage_output = ['F','F','F','F','F','F','F','F','F','F','F','F','F','F','F','F'];
%voltage_output = ['F',num2str(0),num2str(0),'F','F','F','F','F','F','F','F','F','F','F','F','F']; % confirm with Pat how do you update the Voltages
% -4 to +4 or F

active_electrode1 = 1; % the electrode you are actuating



ref_electrode_1 = 2; % the reference electrode



% -1 if the response is proportional and +1 if it is inverse
sign_flag = -1;

aaa = u(1);
set_output(ref_electrode_1,0);
set_output(active_electrode1,aaa);  % send control input to Rasp
update_output();  % send control input to Rasp
u_app(1) = aaa;
delay2 = 10;
pause(delay2)

amp_cell{1} = read_current();
temp_current(:,1) = str2num(amp_cell{1,1}{:});
x = temp_current(active_electrode1,1);
pause(5)
tic
%% Main Loop



for i1=2:len1
    

    x
    %%
    
    amp_cell{i1} = read_current();

    temp_current(:,i1) = str2num(amp_cell{1,i1}{:});
    x(1,i1) = temp_current(active_electrode1,i1);
    e(i1)= x(i1)-x_d(i1);
    
    S(i1) = K_pos*(e(i1)) + (((x(i1)-x(i1-1))/T_samp)-((r(i1)-r(i1-1))/T_samp));
    
    nu(i1) = sign_flag*ro_gain*sign(S(i1)*A_max*cos(u(i1-1)));
%     nu(i1) = sign_flag*ro_gain*sign(S(i1));
    
%    u(i1) = (nu(i1-1) + nu(i1))/T_samp;  % double check maybe change the numerical integral
   u(i1) = u(i1-1)+(nu(i1-1) + nu(i1))/2*T_samp;  
   u_sat(i1) = A_max*sin(u(i1));
     
%      u_sat(i1) = u(i1);
    
    aaa = u_sat(i1); % for closed-loop
    
    
    if aaa>A_max
        aaa = A_max; % for closed-loop
    elseif aaa<min_val
        aaa = min_val;
    end
    
    u_app(i1) = aaa;
    
%      aaa = x_d(i1); % for open-loop
    
set_output(active_electrode1,aaa);  % send control input to Rasp
update_output();  % send control input to Rasp
% amp_cell{i1} = read_current();
% 
% temp_current(:,i1) = str2num(amp_cell{1,i1}{:});


    
    d(i1) = x_d(i1);
    
    subplot(2,3,1)
    plot(d(1:end),'b','linewidth',1.5)
    hold on
    plot(x(2:end)','-or','linewidth',1.5)
    title('Output');
    %legend('Desired Output','SMC-Based Estimated Output','Location','southeast')
    xlabel('Time')
    ylabel('Mean of all pixels')
    grid on
    
    
    subplot(2,3,2)
    plot(u_sat(1:end),'-*g','linewidth',1.5)
    title('Actual Control Output');
    %  legend('Actual Control Effort #1','Location','southeast')
    xlabel('Time')
    ylabel('Voltage (V)')
    grid on
    
    
%     subplot(2,2,3)
%     plot(S(1:end),'-*m','linewidth',1.5)
%     title('plot S');
%     %legend('Scaled Control Effort #1','Location','southeast')
%     xlabel('Time')
%     ylabel('Voltage (V)')
%     
%     grid on


    subplot(2,3,3)
    plot(temp_current(active_electrode1,1:end),'-*m','linewidth',1.5)
    title('Current');
    %legend('Scaled Control Effort #1','Location','southeast')
    xlabel('Time')
    ylabel('Current (nA)')
    
    grid on
    
    
    subplot(2,3,4)
    plot(e(1:end),'-*c','linewidth',1.5)
    title('Tracking Error');
    % legend('Error','Location','southeast')
    xlabel('Time')
    %ylabel('Current (nA)')
    grid on
    
    
    subplot(2,3,5)
    plot(S(1:end),'-*c','linewidth',1.5)
    title('Plot S');
    % legend('Error','Location','southeast')
    xlabel('Time')
    %ylabel('Current (nA)')
    grid on
    
    
    
    subplot(2,3,6)
    plot(u_app(1:end),'-*g','linewidth',1.5)
    title('Applied Control Output');
    %  legend('Actual Control Effort #1','Location','southeast')
    xlabel('Time')
    ylabel('Voltage (V)')
    grid on
    
    hold off
    MM(i1) = getframe(gcf);
    pause(delay)
    
end

toc
close_connection()