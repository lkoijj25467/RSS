
clear;
clc;
%%%%%%%%��������
Pld0 = 41.5;% ����d0(1m)��·�����
a = 2;      % ·���������
Pt = -15;% ���书�ʣ���λdBm

%%%%%%%%   ��200m*200m�ļ�������ھ��ȷֲ�121���ο���
[X,Y] = meshgrid(0:10:200,0:10:200);
dm=[X(:) Y(:)];   %ÿ���ο��������

%%%%%%%%ѡȡ16��ê�ڵ㣬���ȷֲ��ڼ������
RefNode = [25 25;75 25;125 25;175 25;25 75;75 75;125 75;175 75;25 125;75 125;125 125;175 125;25 175;75 175;125 175;175 175]; 
[sizex sizey] = size(RefNode);  % [sizex sizey]=[16,2] 

%%%%%%%%����ָ�ƿ⣬Ҳ���ɼ�ÿ���ο��㵽ê�ڵ��RSSIֵ
 for i=1:size(dm, 1)    %�ο����Ϲ��� 11*11=121����  
        v=randn(sizex,5);   %�����ֵΪ5dbm������
        for j = 1:sizex     %ÿ��ê�ڵ�
            d(i,j)=sqrt((dm(i,1)-RefNode(j,1))^2 + (dm(i,2)-RefNode(j,2))^2);
            fgpt(i,j) = Pt - ( Pld0+10*a*log10(d(i,j)))+v(j); % %fingerprint ָ�ƿ��ڣ����ο��㵽��ê�ڵ���ź�ǿ��   
        end
 end
 
%%%%%%%%ʵ������·��
T=1;%����ʱ��Ϊ1s
pathNode=zeros(200,2); %Ԥ����200��2�еĿվ���,���Ŀ����˶�λ�õ�
nodeV=zeros(200,2);    %Ԥ����200��2�еĿվ��󣬴��Ŀ����˶��ٶ�
nodeA=zeros(200,2);    %Ԥ����200��2�еĿվ��󣬴��Ŀ��ļ��ٶ�
pathNode(1,:)=[10 0];  %��ʼλ��
nodeV(1,:)=[0 0];      %��ʼ�ٶ�
for i=2:200
    nodeA(i,1)=-0.025+50*cos(2*pi*1/105*(i-22))/300;% x����ٶ�
    nodeA(i,2)=cos(2*pi*1/500*i)^2/60;              % y����ٶ�
    nodeV(i,:)=nodeV(i-1,:)+nodeA(i,:)*T;
    pathNode(i,:)=pathNode(i-1,:)+nodeV(i,:)*T+1/2*nodeA(i,:)*T^2;
end
%i=2:2:200;plot(i,nodeA(i,1));grid on;hold on;figure  %��x����ٶ�
%i=2:2:200;plot(i,nodeA(i,2),'--b');grid on;hold on;figure %��y����ٶ�

[mk,nk] = size(pathNode);
%%%%ȷ��·���ϸ����RSSIֵ��Ҳ������ģ�Ͳ���
for kk = 1:mk    %ÿ������λ��Ŀ���
    x0=pathNode(kk,1); % ������  
    y0=pathNode(kk,2); % ������
    v1=randn(sizex,3);%�����ֵΪ3dbm������ 
    for k=1:sizex
        r(k) = sqrt((x0-RefNode(k,1))^2 + (y0-RefNode(k,2))^2); % ����ê�ڵ�ľ���
        online_rssi(kk,k) = Pt - ( Pld0+10*a*log10(r(k)))+v1(k);  %�۲�㵽��ê�ڵ���ź�ǿ��
    end
end

xywknn=zeros(mk,2);
WKNN_wucha=zeros(mk,1);
for kk=1:1:mk  %��ÿ�����Ե�
    [xwknn,ywknn,xknn,yknn]=WKNN(4,fgpt,dm,online_rssi(kk,:));
    
    xywknn(kk,1) = xwknn;
    xywknn(kk,2) = ywknn;
    rmsexwknn =(xywknn(kk,1) - pathNode(kk,1))^2;
    rmseywknn =(xywknn(kk,2) - pathNode(kk,2))^2;
    WKNN_wucha(kk,:)=sqrt(rmsexwknn+rmseywknn);    %����λ����ʵ���˶�λ�õ����
end;    
ave_Err_WKNN=mean(WKNN_wucha,1) %����������ֵ 
var_WKNN=var(WKNN_wucha,0,1);%��������������
[Max_WKNN,xx]=max(WKNN_wucha); %��������

 
i=1:size(dm, 1);plot(dm(i,1),dm(i,2),'*k');hold on;  %�ο���λ��
i=1:16;plot(RefNode(i,1),RefNode(i,2),'^b');hold on; %ê�ڵ�λ��
i=1:2:200;plot(pathNode(:,1),pathNode(:,2),'-k');hold on;  %�˶���λ��
T=1:2:200;plot(xywknn(:,1),xywknn(:,2) ,'--r','linewidth',2,'markersize',4);hold on; %�������λ��

grid on;hold on;


 