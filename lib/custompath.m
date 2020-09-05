function custompath(addTop,addBuild,addLib,addScript)
% custompath([addTop],[addBuild],[addLib],[addScript])
%
% Adds the appropriate folders to Octave's path so that files can be run in
% other directories and still have access to functions.
%
% All arguments are optional. They determine which specific subfolders are
% added. All default values are true.
%
% This file should be edited for any repo-specific external paths.

% Figure out arguments.
if ~exist('addTop','var') || isempty(addTop)
    addTop = true();
end
if ~exist('addScript','var') || isempty(addScript)
    addScript = true();
end
if ~exist('addBuild','var') || isempty(addBuild)
    addBuild = true();
end
if ~exist('addLib','var') || isempty(addLib)
    addLib = true();
end

% Now do path stuff.
thisdir = fileparts(mfilename('fullpath'));
dirs = {'/..',addTop; '',addLib; '/../script',addScript; '/../build',addBuild};
for i = 1:size(dirs,1)
    if dirs{i,2}
        addpath([thisdir,dirs{i,1}])
    end
end

end%function
