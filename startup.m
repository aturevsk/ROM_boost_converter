% startup.m — Run this before using any MATLAB scripts or Simulink models.
% Adds all repository folders to the MATLAB path.

repoRoot = fileparts(mfilename('fullpath'));
addpath(repoRoot);
addpath(fullfile(repoRoot, 'build'));
addpath(fullfile(repoRoot, 'matlab_neural_ss'));
addpath(fullfile(repoRoot, 'simulink_models'));
addpath(fullfile(repoRoot, 'model_data'));
addpath(fullfile(repoRoot, 'data'));

fprintf('ROM_boost_converter paths added.\n');
fprintf('To compare models, run: setup_comparison_profile\n');
