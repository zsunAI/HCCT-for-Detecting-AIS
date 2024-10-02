clc;clear;close all;


path_orginal = fullfile('E:\enhance\data2\centerA\1mm_AI_NCCT');
documentation = dir(fullfile(path_orginal, '*.nii.gz'));
fileNames = {documentation.name};

for n = 1:125
    fileName = fileNames{1, n};
    bgFile = fileName;
    a = fullfile(path_orginal, bgFile);
    a_nii = load_nii(a);
    CT_altas_brain_img0 = a_nii.img;
    

    X = double((CT_altas_brain_img0 - min(CT_altas_brain_img0(:))) / (max(CT_altas_brain_img0(:)) - min(CT_altas_brain_img0(:))));
    width_original = max(CT_altas_brain_img0(:)) - min(CT_altas_brain_img0(:));
    CT_altas_brain_img0 = X;
    
    %% other method used
    enhanced_histeq = histeq(CT_altas_brain_img0);  % histq
    enhanced_clahe = adapthisteq(CT_altas_brain_img0);  % CLAHE
    enhanced_stretch = imadjust(CT_altas_brain_img0, stretchlim(CT_altas_brain_img0), []); % stre
    
    % avg
    enhanced_combined = 0.33 * enhanced_histeq + 0.33 * enhanced_clahe + 0.34 * enhanced_stretch;
    
    % Daubechies
    [x, y, z] = size(CT_altas_brain_img0);
    temp = zeros(x, y, z);
    for a = 1:z
        img1 = enhanced_combined(:, :, a);
        
        img1 = medfilt2(img1);
        
        [c, s] = wavedec2(img1, 3, 'db1');
        
        % same 
        recon_a1 = wrcoef2('a', c, s, 'db1', 1);
        recon_a2 = wrcoef2('a', c, s, 'db1', 2);
        recon_a3 = wrcoef2('a', c, s, 'db1', 3);
        
        % merge
        recon_img3 = 0.3 * recon_a1 + 0.3 * recon_a2 + 0.4 * recon_a3;
        recon_img3 = mat2gray(recon_img3);
        
        % 
        edges = edge(recon_img3, 'sobel');
        recon_img3 = imoverlay(recon_img3, edges, [1 0 0]);  
        
        temp(:, :, a) = recon_img3;
    end
    
    % again histq
    temp1 = histeq(temp);
    
    %% save
    center = 0.9;
    width = 0.2;
    M = mat2gray(temp1, [center - (width / 2), center + (width / 2)]);
    A = struct('hdr', a_nii.hdr, 'img', M);
    outpath_orginal = fullfile('E:\enhance\data2\centerA\processed');
    if ~exist(outpath_orginal, 'dir')
        mkdir(outpath_orginal);
    end
    b = fullfile(outpath_orginal, bgFile);
    save_nii(A, b);
end
