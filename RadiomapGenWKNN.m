
clear;
clc;
%%%%%%%%参数设置
Pld0 = 41.5;% 经过d0(1m)的路径损耗
a = 2;      % 路径损耗因子
Pt = -15;% 发射功率，单位dBm

%%%%%%%%   在200m*200m的监控区域内均匀分布121个参考点
[X,Y] = meshgrid(0:10:200,0:10:200);
dm=[X(:) Y(:)];   %每个参考点的坐标

%%%%%%%%选取16个锚节点，均匀分布于监控区域
RefNode = [25 25;75 25;125 25;175 25;25 75;75 75;125 75;175 75;25 125;75 125;125 125;175 125;25 175;75 175;125 175;175 175]; 
[sizex sizey] = size(RefNode);  % [sizex sizey]=[16,2] 

%%%%%%%%建立指纹库，也即采集每个参考点到锚节点的RSSI值
 for i=1:size(dm, 1)    %参考点上共有 11*11=121个点  
        v=randn(sizex,5);   %加入均值为5dbm的噪声
        for j = 1:sizex     %每个锚节点
            d(i,j)=sqrt((dm(i,1)-RefNode(j,1))^2 + (dm(i,2)-RefNode(j,2))^2);
            fgpt(i,j) = Pt - ( Pld0+10*a*log10(d(i,j)))+v(j); % %fingerprint 指纹库内，各参考点到各锚节点的信号强度   
        end
 end
 
%%%%%%%%实际行走路径
T=1;%采样时间为1s
pathNode=zeros(200,2); %预定义200行2列的空矩阵,存放目标的运动位置点
nodeV=zeros(200,2);    %预定义200行2列的空矩阵，存放目标的运动速度
nodeA=zeros(200,2);    %预定义200行2列的空矩阵，存放目标的加速度
pathNode(1,:)=[10 0];  %起始位置
nodeV(1,:)=[0 0];      %初始速度
for i=2:200
    nodeA(i,1)=-0.025+50*cos(2*pi*1/105*(i-22))/300;% x轴加速度
    nodeA(i,2)=cos(2*pi*1/500*i)^2/60;              % y轴加速度
    nodeV(i,:)=nodeV(i-1,:)+nodeA(i,:)*T;
    pathNode(i,:)=pathNode(i-1,:)+nodeV(i,:)*T+1/2*nodeA(i,:)*T^2;
end
%i=2:2:200;plot(i,nodeA(i,1));grid on;hold on;figure  %画x轴加速度
%i=2:2:200;plot(i,nodeA(i,2),'--b');grid on;hold on;figure %画y轴加速度

[mk,nk] = size(pathNode);
%%%%确定路径上各点的RSSI值，也是利用模型产生
for kk = 1:mk    %每个待定位的目标点
    x0=pathNode(kk,1); % 横坐标  
    y0=pathNode(kk,2); % 纵坐标
    v1=randn(sizex,3);%加入均值为3dbm的噪声 
    for k=1:sizex
        r(k) = sqrt((x0-RefNode(k,1))^2 + (y0-RefNode(k,2))^2); % 到各锚节点的距离
        online_rssi(kk,k) = Pt - ( Pld0+10*a*log10(r(k)))+v1(k);  %观测点到各锚节点的信号强度
    end
end

xywknn=zeros(mk,2);
WKNN_wucha=zeros(mk,1);
for kk=1:1:mk  %对每个测试点
    [xwknn,ywknn,xknn,yknn]=WKNN(4,fgpt,dm,online_rssi(kk,:));
    
    xywknn(kk,1) = xwknn;
    xywknn(kk,2) = ywknn;
    rmsexwknn =(xywknn(kk,1) - pathNode(kk,1))^2;
    rmseywknn =(xywknn(kk,2) - pathNode(kk,2))^2;
    WKNN_wucha(kk,:)=sqrt(rmsexwknn+rmseywknn);    %估算位置与实际运动位置的误差
end;    
ave_Err_WKNN=mean(WKNN_wucha,1) %求列向量均值 
var_WKNN=var(WKNN_wucha,0,1);%求列向量均方差
[Max_WKNN,xx]=max(WKNN_wucha); %求最大误差

 
i=1:size(dm, 1);plot(dm(i,1),dm(i,2),'*k');hold on;  %参考点位置
i=1:16;plot(RefNode(i,1),RefNode(i,2),'^b');hold on; %锚节点位置
i=1:2:200;plot(pathNode(:,1),pathNode(:,2),'-k');hold on;  %运动点位置
T=1:2:200;plot(xywknn(:,1),xywknn(:,2) ,'--r','linewidth',2,'markersize',4);hold on; %画估算的位置

grid on;hold on;


 