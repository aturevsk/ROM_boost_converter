function [dlXOut] = labelWithPropagatedFormats(dlX, dlXRank, formats)
% Function to determine the label a dlarray with the propagated format.
% The U-labelled dlarray is assumed to be in reverse python format. Permute
% to DLT format and label. 
% Since batch is optional in PyTorch, we use rank with propagated label to 
% determine the format and permutations 

%   Copyright 2025 The MathWorks, Inc.

if all(dims(dlX) =='U')
     if contains(formats, "*") && dlXRank == strlength(string(formats))
       formats = replace(formats, "*", "B");
    else
       formats = erase(formats, "*");
     end

    switch formats
        case {'CSS'}
            permReversePythonToDLT       = [2 1 3];
            outputFormat                 = 'SSC';
        case {'BCSS'}
            permReversePythonToDLT       = [2 1 3 4];
            outputFormat                 = 'SSCB';
        case {'CSSS'}
            permReversePythonToDLT       = [3 2 1 4];
            outputFormat                 = 'SSSC';
        case {'BCSSS'}
            permReversePythonToDLT       = [3 2 1 4 5];
            outputFormat                 = 'SSSCB';
        case {'BC'}
            permReversePythonToDLT       = [1 2];
            outputFormat                 = 'CB';
        case {'BCT'}
            permReversePythonToDLT       = [2 3 1]; %Pytorch: BCT || Rev-Pytorch: TCB
            outputFormat                 = 'CBT';
        case {'CT'}
            permReversePythonToDLT       = [2 1];   %Pytorch: CT || Rev-Pytorch: TC
            outputFormat                 = 'CT';
        case {'BTC'}
            permReversePythonToDLT       = [1 3 2]; % PyTorch: BTC || Rev-PyTorch: CTB
            outputFormat                 = 'CBT';
        case {'TC'}
            permReversePythonToDLT       = [1 2]; % PyTorch: TC || Rev-PyTorch: CT
            outputFormat                 = 'CT';
        case {'TBC'}
            permReversePythonToDLT       = [1 2 3]; % PyTorch: TBC || Rev-PyTorch: CBT
            outputFormat                 = 'CBT';
        case {'BT'}
            permReversePythonToDLT       = [2 1]; %PyTorch: BT || Rev-PyTorch: TB
            outputFormat                 = 'BT';
        case {'T'}
            permReversePythonToDLT = [1 2]; %PyTorch: T || Rev-PyTorch: T
            outputFormat = 'TU';
        otherwise
            error('Unknown format and rank %s', formats);
    end
    
    dlXPermuted = permute(stripdims(dlX),permReversePythonToDLT);
    dlXOut = dlarray(dlXPermuted,outputFormat);
    % dlXOut.rank = dlXRank;
else
    dlXOut = dlX;
end

end
