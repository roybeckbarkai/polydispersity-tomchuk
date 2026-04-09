%% Initialize
req_vars = {'instrument', 'simulation', 'ensemble', 'dnames'};
% Force the check to happen in the 'caller' workspace (the script)
missing = cellfun(@(x) ~evalin('caller', ['exist(''', x, ''', ''var'')']), req_vars);

if any(missing)
    fprintf('initializing all params\n')
    [instrument, simulation, ensemble, dnames] = init_TENOR_params()
end
%% get measurement:
meas_source='simulate'; % source can be 'simulate' to simulate, or a struct with I(q)
if strcmpi(meas_source,'simulate')
    % Simulate test case
    if ~exist('PhotLevel','var'), PhotLevel=0; end  % if negative means the number of photons in the central pixel, 0 is no noise.
    if ~exist('V','var'), V=0.1; end
    [qx, qy, I_noisy, ~] = Scatter2D(ensemble.rg, PhotLevel, V, ...
        ensemble.nu, instrument.DETpix, instrument.SD_dist, ...
        instrument.lambda, instrument.det_side, 1, ...
        ensemble.d_nam, ensemble.dist_param, ensemble.Scatter_R_g_weight);
    dq=4*pi/instrument.lambda*instrument.det_side/instrument.SD_dist/(2*round(instrument.DETpix/2)+1); % q resolution
    if exist('save_dir','var') % save simulation result
        sim_save(save_dir, qx, qy, I_noisy, V, instrument, ensemble)
    end
    %% read sequence
    % qx = h5read(filename, '/qx');
    % qy = h5read(filename, '/qy');
    % I_mat = h5read(filename, '/I_mat');

    %% python read sequence
    % import h5py
    %
    % with h5py.File('measurement_data.h5', 'r') as f:
    %     # Use .astype(float) if you want to ensure double precision
    %     qx = f['qx'][:]
    %     qy = f['qy'][:]
    %     I_mat = f['I_mat'][:]
    %
    %     # IMPORTANT: MATLAB is Column-major, Python is Row-major.
    %     # If the orientation matters for your math, transpose them:
    %     # qx = qx.T


elseif isstruct(meas_source)
    try
        qx=meas_source.q_matx;
        qy=meas_source.q_maty;
        I_noisy=meas_source.I_mat;
        dq = abs(mean(diff(sort(unique(qy)))));
    catch
        fprintf ('source measurement not found\n')
    end
end

%% Analyze measurement
if ~isstruct(GT_lib), % The formerly calculated ground truth
    GT_lib.V_list = [];
    GT_lib.RgMeas_list = [];
    GT_lib.Yg100_list = [];
    GT_lib.Yg210_list = [];
    GT_lib.Ym210_list = [];
    GT_lib.RgTrue_covered = [];
    %     load GroundTruth_sphere_rg5_lognormal.mat
end
[V_rec, Sols, Winner, GT_lib, rg_in, Yg100, Yg210, Ym210] = Tenor_Process_Landscape(I_noisy, qx, qy, GT_lib, instrument, simulation, ensemble);
Results_entry.Noise = PhotLevel;
Results_entry.True_V = V;
Results_entry.Primary_V = V_rec;
Results_entry.Winner = Winner;
Results_entry.Rg_meas = rg_in;
Results_entry.Yg100 = Yg100;
Results_entry.Yg210 = Yg210;
Results_entry.Ym210 = Ym210;
Results_entry.V_all = [V_rec Sols.Alternatives(:)'];
altern=[V_rec Sols.Alternatives(:)'];
altern=altern(altern>-0.05);
if numel(altern)>0
    [~,best]=min(abs(altern-V));
    best=altern(best);
else
    best=nan;
end



fprintf('Phot dens= %0.2e ph*nm^2 | V= %.3f | nearest V= %.3f | V discrep= %.4f\n',-PhotLevel/dq^2, V, best, best- V);
%%

function sim_save(save_dir_raw, qx, qy, I_noisy, V, instrument, ensemble)
    % 1. Resolve the Windows .lnk to a real path
    resolved_dir = resolve_shortcut(save_dir_raw);
    
    % Create directory if it doesn't exist
    if ~exist(resolved_dir, 'dir')
        mkdir(resolved_dir); 
    end
    
    % 2. Find the next consecutive index
    existing = dir(fullfile(resolved_dir, 'sim_*_*.h5'));
    if isempty(existing)
        next_idx = 1;
    else
        % Extract index from 'sim_0001_V0.100.h5'
        tokens = regexp({existing.name}, 'sim_(\d+)_', 'tokens');
        % Remove empty matches and extract values
        valid_tokens = tokens(~cellfun(@isempty, tokens));
        indices = cellfun(@(x) str2double(x{1}{1}), valid_tokens);
        next_idx = max(indices) + 1;
    end
    
    % 3. Setup Filenames
    fname_base = sprintf('sim_%04d_V%0.3f', next_idx, V);
    final_h5_path = fullfile(resolved_dir, [fname_base, '.h5']);
    temp_h5_path = fullfile(tempdir, [fname_base, '.h5']); % Local Temp Storage
    csv_name = fullfile(resolved_dir, 'sim_metadata_log.csv');
    
    % 4. Save HDF5 (Locally to avoid Google Drive Sync locks)
    data_size = size(I_noisy);
    chunk = [min(100, data_size(1)), min(100, data_size(2))];
    
    try
        % Ensure no old temp file exists
        if exist(temp_h5_path, 'file'), delete(temp_h5_path); end
        
        h5create(temp_h5_path, '/qx', data_size, 'ChunkSize', chunk, 'Deflate', 5);
        h5create(temp_h5_path, '/qy', data_size, 'ChunkSize', chunk, 'Deflate', 5);
        h5create(temp_h5_path, '/I_noisy', data_size, 'ChunkSize', chunk, 'Deflate', 5);
        
        h5write(temp_h5_path, '/qx', qx);
        h5write(temp_h5_path, '/qy', qy);
        h5write(temp_h5_path, '/I_noisy', I_noisy);
        
        % 5. Move finished file to Google Drive
        [status, msg] = movefile(temp_h5_path, final_h5_path, 'f');
        if ~status
            error('Could not move file to Google Drive: %s', msg);
        end
        
    catch ME
        if exist(temp_h5_path, 'file'), delete(temp_h5_path); end
        rethrow(ME); % Pass the error up so you know what happened
    end
    
    % 6. Metadata Logging
    write_header = ~exist(csv_name, 'file');
    fid = fopen(csv_name, 'a');
    if fid == -1
        warning('Could not open CSV log file. Check if it is open in Excel.');
    else
        if write_header
            fprintf(fid, 'Date,Index,Filename,V,Rg,DistType,Lambda,SD_cm\n');
        end
        
        t_str = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        fprintf(fid, '%s,%d,%s,%.4f,%.2f,%s,%.3f,%.1f\n', ...
            t_str, next_idx, [fname_base, '.h5'], V, ...
            ensemble.rg, ensemble.d_nam, instrument.lambda, instrument.SD_dist);
        fclose(fid);
    end
    
    fprintf('Saved %s to ...%s\t', fname_base, resolved_dir(max(1,end-25):end));
end
%%
function real_path = resolve_shortcut(input_path)
    % Find where the .lnk extension is
    idx = strfind(input_path, '.lnk');
    if isempty(idx)
        real_path = input_path;
        return;
    end
    
    % Split into: [Path_to_link.lnk] and [\Subfolder]
    lnk_part = input_path(1:idx+3);
    sub_part = input_path(idx+4:end);
    
    % Get the absolute path of the .lnk file
    file_info = dir(lnk_part);
    if isempty(file_info)
        error('The shortcut file %s does not exist.', lnk_part);
    end
    abs_lnk = fullfile(file_info.folder, file_info.name);
    
    % Use Windows COM to read the shortcut target
    shell = actxserver('WScript.Shell');
    shortcut = shell.CreateShortcut(abs_lnk);
    target_base = shortcut.TargetPath;
    release(shell); % Clean up COM object
    
    % Combine the target with the remaining subfolder
    real_path = fullfile(target_base, sub_part);
end