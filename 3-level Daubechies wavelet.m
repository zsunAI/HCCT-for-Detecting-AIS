clc;clear;close all;
path_orginal=fullfile('E:\enhance\data2\centerA\1mm_AI_NCCT');
% path_orginal=fullfile('E:\enhance\data\001_NII_SITE005_PVB_NCCT\step4_1mm_for_AI_Thin_slice\1mm');
documentation=dir(fullfile(path_orginal,'*.nii.gz'));

fileNames = {documentation.name};
for n = 1:125 %  125 is number of patients in floders
   fileName = fileNames{1,n};
   bgFile = fileName;
   a = [path_orginal,'\',bgFile]; 
   a_nii = load_nii(a);
   CT_altas_brain_img0 = a_nii.img;
   % CT_altas_brain_img01 = CT_altas_brain_img0(:,:,50);
   % as = imhist(CT_altas_brain_img01);
   % figure,plot(as);
   X =    double( ( CT_altas_brain_img0-min(CT_altas_brain_img0(:)) )/ (max(CT_altas_brain_img0(:))-min(CT_altas_brain_img0(:)) ) );
  
   width_original = max(CT_altas_brain_img0(:))-min(CT_altas_brain_img0(:));
   CT_altas_brain_img0 = X;
   [x,y,z] = size(CT_altas_brain_img0);
   initial = CT_altas_brain_img0;
   new = histeq(initial); 
   temp = zeros(x,y,z);
   temp2 = zeros(x,y,z);
   temp12 = zeros(x,y,z);
   temp1  = zeros(x,y,z);
   for a = 1:z
        img1 = new(:,:,a);
        % figure,imshow(img1,[]);
        [c,s]=wavedec2(img1,3,'db1');% -db1
        % [cH2,cV2,cD2]=detcoef2('all',c,s,2); high
        % cA1=appcoef2(c,s,'db1',1);
        % cA2=appcoef2(c,s,'db1',2);
    
        %% 01 
        a_ca1= appcoef2(c,s,'db1',1); % low -freq
        a_ch1= detcoef2('h',c,s,1); % high _h  
        a_cv1= detcoef2('v',c,s,1); % high _V  
        a_cd1= detcoef2('d',c,s,1); % high _D  
        %% 02
        a_ca2= appcoef2(c,s,'db1',2); %
        a_ch2= detcoef2('h',c,s,2); % high _h
        a_cv2= detcoef2('v',c,s,2); % high _V
        a_cd2= detcoef2('d',c,s,2); % high _D
    
        %% 
        a_ca2= appcoef2(c,s,'db1',3); %
        recon_a1 = wrcoef2('a',c,s,'db1',1);%
        recon_a2 = wrcoef2('a',c,s,'db1',2);%
        recon_a3 = wrcoef2('a',c,s,'db1',3);%
    
        recon_h1 = wrcoef2('h',c,s,'db1',1);
        recon_v1 = wrcoef2('v',c,s,'db1',1);
        recon_d1 = wrcoef2('d',c,s,'db1',1);
    
        %% three low freq
        recon_set=[recon_a1, recon_h1;recon_v1,recon_d1]; % test to show low re-freq
        % figure,imshow(recon_set,[]);
        recon_img3 = recon_a1 + recon_a2 + recon_a3; % 

        recon_img3 = mat2gray(recon_img3);

        temp(:,:,a)= recon_img3;

        % figure,imshow(recon_img,[]);
   end
   temp1 = histeq(temp); 
   center = 0.9;
   width = 0.2;

   M=mat2gray(temp1,[center-(width/2),center+(width/2)]); 
   A = struct('hdr',a_nii.hdr,'img',M);
   outpath_orginal=fullfile('E:\enhance\data2\centerA\processed');
   b = [outpath_orginal,'\',bgFile];
   save_nii(A,b);
   
end
