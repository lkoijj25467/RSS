% WKNN算法和KNN算法
% K:最近邻数；rssi_fgpt:所用的指纹库；rssi_dm:指纹库对应的指纹位置； rssi:在线采集的指纹
% 返回的是估计的x,y坐标
function [xwknn,ywknn,xknn,yknn]=WKNN(K,rssi_fgpt,rssi_dm,rssi)
    [t1,t2]=size(rssi_fgpt);
    t3=repmat(rssi,[t1,1]); %观测点的信号强度值复制，形成一个与指纹库一样行的信号强度矩阵，便于同指纹比较
    temp1 =rssi_fgpt - t3;
    wknn=sqrt(sum((temp1.^2),2));
    [LMAX,ROW]=max(wknn(:));
    
    wknnfmt= zeros(K,3);  %存储最邻近指纹点的坐标
    xwknn=0;    % WKNN算法坐标
    ywknn=0;
    wknnsum=0;   
    %获取距离最小的K个匹配网格，并把相应坐标存于knnfmt
    for k = 1:K
        [L,M] = min(wknn(:)); % 返回L为最小值，M为行数
        wknnfmt(k,1)=rssi_dm(M,1);
        wknnfmt(k,2)=rssi_dm(M,2);
        wknnfmt(k,3)=L;
        wknn(M)=LMAX;
        wknnsum=wknnsum+1/L;
    end
    %获取带权重的估算坐标
    for k=1:K
        xwknn=xwknn+1/wknnfmt(k,3)/wknnsum*wknnfmt(k,1);
        ywknn=ywknn+1/wknnfmt(k,3)/wknnsum*wknnfmt(k,2);
    end    
    
    xknn=0;% KNN算法坐标
    yknn=0;
    knnsum=0;   
    %KNN近域算法坐标
    for k=1:K
        xknn=xknn+wknnfmt(k,1);
        yknn=yknn+wknnfmt(k,2);
    end
    xknn= xknn/K;
    yknn= yknn/K;
end