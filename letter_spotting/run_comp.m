clear
%addpath('algorithm', 'general', 'comp',' vlfeat');
%if ~exist('vl_version','file'), run('vlfeat/toolbox/vl_setup.m'); end
addpath('algorithm', 'general', 'comp');
%addpath('algorithm', 'general', 'comp');

seg_based = 'comp/Segmentation-based_';
seg_free  = 'comp/Segmentation-free_';
dataset = 'Dataset';
queries_path = 'Queries';

set = struct('pagesDir', {[seg_free, dataset]},'segDir', {[seg_based, dataset]},...
                       'queriesDir', {[seg_free, queries_path]});
%imgExt = cell(1,1);
%imgExt{1} = '.tif';
%imgExt{2} = '.jpg';
imgExt = '.jpg';
%imgExt = '.png';

pathToDocs = 'Documents/';                                                 % documents directory (inside segmentation free directory)
segFile = '/Segments.mat';
pfile = fullfile(set.pagesDir, 'P.mat');                                       % name for saving/loading the docs struct variable
PPP_file = fullfile(set.pagesDir, 'PPP.mat');
modelFile = fullfile(set.segDir, 'Model.mat');                         % name and location for saving/loading the model struct variable
queriesFile = 'queries.gtp';                                                     %  queries file
cache = 1;

sParam = get_default_cfg();                                                    % setting the default configurations and parameters
sParam.showDebug = 1;

disp(['Running ' set.pagesDir]);

% Creating the model - reading pages, preprocessing, finding candidates, and building the model
% 1. read pages
[docs,dic] = read_pages(set.pagesDir, imgExt, cache, pathToDocs, pfile);

% 2. preprocessing pages
PPP = pre_process_pages2(docs,sParam,set.pagesDir,cache,PPP_file);

% 3. read/build segments and create a model
Segments = build_windows_parfor(PPP, sParam,set.segDir,segFile,cache);
[Candidates, AllCandidates]=prepare_segments (Segments);
save_image_candidates(Candidates, docs,PPP,set.segDir);

% DEBUGGING - saving the documents with markings of the bounding boxes.
% (it will be saved in the segments directory, under a directory named 'Cand', and 'Cand2' for BW inverted)
% draw_squares(Segments,PPP,docs,set.segDir) %UNCOMMENT THIS LINE FOR  DEBUGGING

