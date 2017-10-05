function sParam=get_default_cfg()

%HMAX parameters
sParam.fdesc = 'hog+lbp';                              % hog|lbp|hog+lbp
sParam.CS=8;                                                % Feature descriptor cell size
sParam.H=sParam.CS*8;                               % Resize picture heigth
sParam.W=sParam.CS*20;                            % Resize picture width


sParam.modelTo = 250;                                 % Final vector length   
sParam.modelFrom = sParam.modelTo*15;  % number of random candidates for model building
% sParam.dimension = 500;                           % PCA dimension reduction - Final vector length

sParam.bwMedian = 0.5;                              % 0.85-0.87 Gniza

%CC parameters
sParam.bwWTooBig = 90;                            % Width too big
sParam.bwHTooBig = 165;                             % Height too big
sParam.bwTooBig = 14850;                            % number of pixels too big
sParam.bwTooSmall = 100;                              % number of pixels too small
sParam.HTooSmall = 60;                                % Height too small
sParam.WTooSmall = 40;                               % Width too small

sParam.segMaxWin = 10000;

%Candidate's bounding boxes parameters
sParam.findW = sParam.bwWTooBig;                                       % bounding box's maximum width
sParam.findH = sParam.bwHTooBig;                                        % bounding box's maximum height
sParam.findHMin = sParam.HTooSmall;                                    % bounding box's minimum heigth
sParam.findWMin = sParam.WTooSmall;                                   % bounding box's maximum width
sParam.findT = 6;                                          % maximum space between lowest and heighest CC in the bounding box

sParam.findSpace=1;                                    % number of pixels for checking space (before, after and inside a word)
sParam.comma=50;

%if equals 1 the requirement for space between words is removed
sParam.noSpace=1;

sParam.improveMrgn=30;
sParam.improvePass=0.7;

 % jittering process parameters
sParam.q_move_i = -8:4:8;
sParam.q_move_j =-16:4:16;
sParam.q_move_k = 4:2:10;

% flag for running/not running re-ranking
sParam.runPostProcessing=500;

sParam.resToSave = 500;                             % number of results to save
sParam.numImagesToSave = 500;               % number of results images to save (must be smaller or equal to resToSave)

sParam.cacheMatFile = '';

sParam.saveWindows = 1;
sParam.showDebug = 1;




