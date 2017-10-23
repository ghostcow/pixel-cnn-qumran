'''
# run script from cli format:
matlab -nodisplay -nosplash -nodesktop -r "try, run('run_comp.m'); catch, exit(1); end; exit;"
'''

'''
sketch - use this script to run letter spotting on each jpg manually for param tuning etc, 
'''

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--bwMedian', type=float, default=8.0,
                    help='threshold at value ( bwMedian * <median value of the image> ) (0.85-0.87 Gniza)')
# ConnectedComponents parameters
parser.add_argument('--bwWTooBig', type=int, default=350,
                    help='CC Width too big')
parser.add_argument('--bwHTooBig', type=int, default=400,
                    help='CC Height too big')
parser.add_argument('--bwTooBig', type=int, default=71000,
                    help='number of pixels in CC too big')
parser.add_argument('--bwTooSmall', type=int, default=10,
                    help='number of pixels in CC too small')
parser.add_argument('--HTooSmall', type=int, default=40,
                    help='CC Height too small')
parser.add_argument('--WTooSmall', type=int, default=40,
                    help='CC Width too small')
parser.add_argument('--segMaxWin', type=int, default=10000,
                    help='')
# Candidate's bounding box parameters
parser.add_argument('--findW', type=int, default=350,
                    help='bounding box maximum width')
parser.add_argument('--findH', type=int, default=400,
                    help='bounding box maximum height')
parser.add_argument('--findHMin', type=int, default=40,
                    help='bounding box minimum height')
parser.add_argument('--findWMin', type=int, default=40,
                    help='bounding box minimum width')
parser.add_argument('--findT', type=int, default=15,
                    help='maximum space between lowest and highest connected component in the bounding box')
# Miscellaneous
parser.add_argument('--debug', action='store_true',
                    help='Debug mode')
args = parser.parse_args()

print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args


default_config_fn = '''function sParam=get_default_cfg()

sParam.bwMedian = {bwMedian};                              % threshold at value ( bwMedian * <median value of the image> ) (0.85-0.87 Gniza)

%CC parameters
sParam.bwWTooBig = {bwWTooBig};                     % Width too big
sParam.bwHTooBig = {bwHTooBig};                     % Height too big
sParam.bwTooBig = {bwTooBig};                    % number of pixels too big
sParam.bwTooSmall = {bwTooSmall};                     % number of pixels too small
sParam.HTooSmall = {HTooSmall};                      % Height too small
sParam.WTooSmall = {WTooSmall};                      % Width too small

sParam.segMaxWin = {segMaxWin};

%Candidate's bounding boxes parameters
sParam.findW = {findW};            % bounding box's maximum width
sParam.findH = {findH};            % bounding box's maximum height
sParam.findHMin = {findHMin};         % bounding box's minimum heigth
sParam.findWMin = {findWMin};         % bounding box's maximum width
sParam.findT = {findT};                          % maximum space between lowest and highest CC in the bounding box

sParam.findSpace=1;                         % number of pixels for checking space (before, after and inside a word)
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
sParam.resToSave = 500;                     % number of results to save
sParam.numImagesToSave = 500;               % number of results images to save (must be smaller or equal to resToSave)
sParam.cacheMatFile = '';
sParam.saveWindows = 1;
sParam.showDebug = 1;
'''

# back up previous cfg
import shutil
shutil.move('letter_spotting/algorithm/get_default_cfg.m',
            'letter_spotting/algorithm/get_default_cfg.m.bak')

# write out cfg for current run
with open('letter_spotting/algorithm/get_default_cfg.m', 'w') as f:
    f.write(default_config_fn.format(**vars(args)))

# call run_comp
import os
import subprocess
os.chdir('letter_spotting')
if args.debug:
    cmd = '''try, run('run_comp.m'); catch ME, rethrow(ME); exit(1); end; exit;'''
else:
    cmd = '''try, run('run_comp.m'); catch, exit(1); end; exit;'''
subprocess.call(['matlab', '-nodisplay', '-nosplash', '-nodesktop', '-r', cmd])
os.chdir('..')
