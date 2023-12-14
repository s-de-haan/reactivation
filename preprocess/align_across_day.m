%% open masks from python 
% mice = {'NN8', 'NN9', 'NN11'};
% dates_NN8 = {'210312', '210314', '210316', '210318', '210320', '210321', '210327', '210329', '210330'};
% dates_NN9 = {'210428', '210429', '210501', '210502', '210503', '210505', '210506', '210507', '210509', '210510', '210511', '210512', '210513', '210514'};
% dates_NN11 = {'210626', '210627', '210628', '210629', '210630', '210701', '210703', '210704', '210705', '210706'};
% mice_dates = {dates_NN8, dates_NN9, dates_NN11};

mouse = 'NN11';
date_base = '210627';
date_align = '210628';

cell_masks_plane_1 = readNPY(['D:\2p_data\scan\',mouse,'\',date_base,'_',mouse,'\processed_data\saved_data\plane_1_cell_masks.npy']);
cell_masks_plane_other = readNPY(['D:\2p_data\scan\',mouse,'\',date_align,'_',mouse,'\processed_data\saved_data\plane_1_cell_masks.npy']);

%% make projection
fixed = sum(cell_masks_plane_1,3);
moving = sum(cell_masks_plane_other,3);
%figure; imshowpair(fixed,moving)
[d,movingReg] = imregdemons(moving,fixed,[200 100 50],'AccumulatedFieldSmoothing',1.0);
figure; imshowpair(fixed,movingReg)

%% apply warp to all individual cell masks
cell_masks_plane_other_reg = imwarp(cell_masks_plane_other, d);

%% see which cells overlap from other plane
cell_masks_plane_1_num = cell_masks_plane_1;
for i = 1:size(cell_masks_plane_1, 3)
    mask = cell_masks_plane_1(:,:,i);
    mask(mask > 0) = i;
    cell_masks_plane_1_num(:,:,i)= mask;
end
fixed_num = sum(cell_masks_plane_1_num,3);
overlap_vec = zeros(size(cell_masks_plane_other_reg,3),3);
for i = 1:size(cell_masks_plane_other_reg, 3)
    cell_plane_other = cell_masks_plane_other_reg(:,:,i) > 0;
    which_plane_1_cell = fixed_num(cell_plane_other);
    [occurance,cells] = findgroups(which_plane_1_cell);
    [occurances,~] = histc(occurance,unique(occurance));
    if ismember(0, cells)
        occurances(cells == 0) = [];
        cells(cells == 0) = [];
    end
    [~,I] = max(occurances);
    cell = cells(I);
    if ~isempty(cell)
        cell_plane_1 = cell_masks_plane_1(:,:,cell);
        cell_plane_other = cell_masks_plane_other_reg(:,:,i);
        temp_sum = cell_plane_1+cell_plane_other;
        temp_sum = temp_sum == 2;
        overlap_1 = sum(sum(temp_sum))/sum(sum((cell_plane_1)));
        if overlap_1 > overlap_vec(i,2)
            overlap_vec(i,2) = overlap_1;
            overlap_vec(i,1) = cell;
            overlap_vec(i,3) = sum(sum(temp_sum))/sum(sum((cell_plane_other)));
        end  
    end
end
         