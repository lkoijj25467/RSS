% WKNN�㷨��KNN�㷨
% K:���������rssi_fgpt:���õ�ָ�ƿ⣻rssi_dm:ָ�ƿ��Ӧ��ָ��λ�ã� rssi:���߲ɼ���ָ��
% ���ص��ǹ��Ƶ�x,y����
function [xwknn,ywknn,xknn,yknn]=WKNN(K,rssi_fgpt,rssi_dm,rssi)
    [t1,t2]=size(rssi_fgpt);
    t3=repmat(rssi,[t1,1]); %�۲����ź�ǿ��ֵ���ƣ��γ�һ����ָ�ƿ�һ���е��ź�ǿ�Ⱦ��󣬱���ָͬ�ƱȽ�
    temp1 =rssi_fgpt - t3;
    wknn=sqrt(sum((temp1.^2),2));
    [LMAX,ROW]=max(wknn(:));
    
    wknnfmt= zeros(K,3);  %�洢���ڽ�ָ�Ƶ������
    xwknn=0;    % WKNN�㷨����
    ywknn=0;
    wknnsum=0;   
    %��ȡ������С��K��ƥ�����񣬲�����Ӧ�������knnfmt
    for k = 1:K
        [L,M] = min(wknn(:)); % ����LΪ��Сֵ��MΪ����
        wknnfmt(k,1)=rssi_dm(M,1);
        wknnfmt(k,2)=rssi_dm(M,2);
        wknnfmt(k,3)=L;
        wknn(M)=LMAX;
        wknnsum=wknnsum+1/L;
    end
    %��ȡ��Ȩ�صĹ�������
    for k=1:K
        xwknn=xwknn+1/wknnfmt(k,3)/wknnsum*wknnfmt(k,1);
        ywknn=ywknn+1/wknnfmt(k,3)/wknnsum*wknnfmt(k,2);
    end    
    
    xknn=0;% KNN�㷨����
    yknn=0;
    knnsum=0;   
    %KNN�����㷨����
    for k=1:K
        xknn=xknn+wknnfmt(k,1);
        yknn=yknn+wknnfmt(k,2);
    end
    xknn= xknn/K;
    yknn= yknn/K;
end