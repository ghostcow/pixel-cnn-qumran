clear
addpath('algorithm', 'general', 'comp',' vlfeat');
if ~exist('vl_version','file'), run('vlfeat/toolbox/vl_setup.m'); end
addpath('algorithm', 'general', 'comp');

seg_based = 'comp/Segmentation-based_';
seg_free   = 'comp/Segmentation-free_';
sets = {'DATASET'};                                          % ENTER HERE THE NAME OF THE DATASET
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
resFile = 'RelevanceRankedList.xml';
resToSave = 500;                                                                % Number of results to save for each query
cache = 1;

sParam=get_default_cfg();                                                    % setting the default configurations and parameters
sParam.showDebug = 1;

disp(['Running ' set.pagesDir]);

% Creating the model - reading pages, preprocessing, finding candidates, and building the model
% 1. read pages
[docs,dic] = read_pages(set.pagesDir, imgExt, cache, pathToDocs, pfile);

% 2. preprocessing pages
PPP = pre_process_pages2(docs,sParam,set.pagesDir,cache,PPP_file);

% 3. read/build segments and create a model
Segments=build_windows_parfor(PPP, sParam,set.segDir,segFile,cache);
[Candidates, AllCandidates]=prepare_segments (Segments);
save_image_candidates(Candidates, docs,PPP,set.segDir);

% 4. save params to file
save(fullfile(set.segDir, 'config.mat'), 'sParam');

% DEBUGGING - saving the documents with markings of the bounding boxes.
% (it will be saved in the segments directory, under a directory named 'Cand', and 'Cand2' for BW inverted)

draw_squares(Segments,PPP,docs,set.segDir) %UNCOMMENT THIS LINE FOR  DEBUGGING

%Load the modelFile if exist, if not continue
% if cache==1 && exist(modelFile, 'file')
%     disp(' - Loading Model file')
%     load(fullfile(modelFile));
% else
% 4. build model matrix
% Model = create_model_build_matrix(PPP,Segments,docs,sParam);

% 5. build pages matrix
% Model = create_model_build_pages(Model,PPP,modelFile,cache,docs,sParam);

% end

%% Reading query words and running word spotting

% 6. read queries
% queries=read_queries2(set.queriesDir, queriesFile,cache);

% 7. run queries
% [Results,scores,QXD] = run_queries2(Model,queries,sParam,resToSave, set.queriesDir,cache);
% 8. save results
% XML=save_results2(Results, queries, docs, Segments, fullfile(set.queriesDir,resFile), sParam.resToSave);

% 9. save results images
% save_images_results(XML, docs,Results,sParam)
% (it will be saved in a directory named 'results')

% 10. creating results image
% numpic = 150; %amount of results in one image
% numLines = 10; %number of result images in one column
% create_result_image(numpic,numLines,queries);



