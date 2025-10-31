%% This function gets the all behavior data for awake+quiet waking 
mice = {'NN8'}; %,'NN9','NN11','NN13','NN16','NN17','NN23','NN28'};
dates_all = {{'210312'}};

for m = 1:size(mice,2)
    mouse = mice{m};
    dates = dates_all{m};
    for dd = 1:size(dates,2)
        date = dates{dd};
        %% task variables
        task_runs_vec = {'2','3','4','5'};
        dark_runs_vec = {'1'};
        task_runs = size(task_runs_vec,2);
        dark_runs = size(dark_runs_vec,2);
        planes = 3;
        frames_per_run = floor(64000/planes);
        ITI = 55;
        framerate = 31.25/planes;
        onsets = [];
        offsets = [];
        US_1_onset = [];
        US_2_onset = [];
        temp_licking = [];
        trial_error = [];
        cue_code = [];
        running = [];
        pupil = [];
        pupil_movement = [];
        licking = [];
        
        %% load behavioral data for dark runs
        for i = 1:size(dark_runs_vec,2)
            run = dark_runs_vec{i};
            % running
            quad_path = ['/Users/sander/PhD/reactivation/andermann_data/',mouse,'/',date,'_',mouse,'/',date,'_',mouse,'_00',run,'/',mouse,'_',date,'_00',run,'_quadrature.mat'];
            quadfile = builtin('load', quad_path, '-mat');
            if ~isempty(quadfile)
                temp_running = double(quadfile.quad_data);
                last = round(size(temp_running,2)/3)*planes;
                nframes = size(temp_running,2);
                temp_running = pipe.misc.position_to_speed(temp_running,framerate);
                temp_running = resample(temp_running,last/planes,size(temp_running,2));
            end
            running = [running temp_running];
            % pupil
            pipe.pupil.masks(mouse,date,run);
            [dx, dy, psum, area, quality] = pipe.pupil.extract(mouse,date,run);
            area = resample(area,last/planes,size(area,2));
            pupil = [pupil area];
            dx = resample(dx,last/planes,size(dx,2));
            dy = resample(dy,last/planes,size(dy,2));
            d = sqrt(dx.^2 + dy.^2);
            pupil_movement = [pupil_movement d];
            %licking
            temp_licking = zeros(1,frames_per_run);
            licking = [licking temp_licking];
        end
        
        %% load behavioral data for task runs
        for i = 1:size(task_runs_vec,2)
            run = task_runs_vec{i};
            %behavior data
            bdata = pipe.io.trial_times(mouse,date,run);
            bdata.onsets = floor(bdata.onsets/planes);
            bdata.offsets = floor(bdata.offsets/planes);
            bdata.ensure = floor(bdata.ensure/planes);
            bdata.quinine = floor(bdata.quinine/planes);
            onsets = [onsets;double(bdata.onsets)+(frames_per_run*(i-1))+(dark_runs*frames_per_run)];
            offsets = [offsets;double(bdata.offsets)+(frames_per_run*(i-1))+(dark_runs*frames_per_run)];
            bdata.ensure = double(bdata.ensure);
            bdata.quinine = double(bdata.quinine);
            bdata.ensure(bdata.ensure > 0) = bdata.ensure(bdata.ensure > 0) + (frames_per_run*(i-1)) +(dark_runs*frames_per_run);
            bdata.quinine(bdata.quinine > 0) = bdata.quinine(bdata.quinine > 0) + (frames_per_run*(i-1)) +(dark_runs*frames_per_run);
            US_1_onset = [US_1_onset;bdata.ensure];
            US_2_onset = [US_2_onset;bdata.quinine];
            trial_error = [trial_error;bdata.trialerror];
            cue_code = [cue_code;bdata.condition];
            
            %running
            temp_running = [];
            quad_path = ['D:\2p_data\scan\',mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_00',run,'\',mouse,'_',date,'_00',run,'_quadrature.mat'];
            quadfile = builtin('load', quad_path, '-mat');
            if ~isempty(quadfile)
                temp_running = double(quadfile.quad_data);
                last = round(size(temp_running,2)/3)*planes;
                nframes = size(temp_running,2);
                temp_running = pipe.misc.position_to_speed(temp_running,framerate);
                temp_running = resample(temp_running,last/planes,size(temp_running,2));
            end
            running = [running temp_running];
            
            %pupil
            if exist(['D:/2p_data/scan/',mouse,'/',date,'_',mouse,'/',date,'_',mouse,'_00',run,'/',mouse,'_',date,'_00',run,'.pdiam'], 'file')
                pipe.pupil.masks(mouse,date,run);
                [dx, dy, psum, area, quality] = pipe.pupil.extract(mouse,date,run);
                area = resample(area,last/planes,size(area,2));
                pupil = [pupil area];
                dx = resample(dx,last/planes,size(dx,2));
                dy = resample(dy,last/planes,size(dy,2));
                d = sqrt(dx.^2 + dy.^2);
                pupil_movement = [pupil_movement d];
            else
                pupil = [pupil zeros(1, size(temp_running,2))];
                pupil_movement = [pupil_movement zeros(1, size(temp_running,2))];
            end
                
            %licking
            temp_licking = zeros(1,nframes);
            temp_licking(bdata.licking) = 1;
            temp_licking = resample(temp_licking,last/planes,size(temp_licking,2));
            licking = [licking temp_licking];
        end
        cue_code(cue_code == 4) = 3; % makes pavs same as conditional
        cue_code(cue_code == 7) = 8; % make cond quinine pav
        CS_1_code = 3; %%%%%%%%%%
        CS_2_code = 8; %%%%%%%%%%
        licking = zeros(1,size(licking,2));
        
        %% save file
        base = 'D:/2p_data/scan/';
        cd ([base,mouse,'\',date,'_',mouse,'\processed_data\saved_data'])
        save('behavior_file')
        
        %% copy over suite2p
        % copy_folders('//nasquatch/data/2p/nghia/','D:/2p_data/scan/',mouse,date,0,3);
        
    end
end
close all; clear all; clc
