function [Zval, Zrank] = pyElementwiseBinary(Xval, Yval, AlphaVal, MatlabFcn, InputRanks)
%Calculates elementise binary 'matlabFcn' for Xval and Yval

%Copyright 2025 The MathWorks, Inc.

import neuralode_mlp_traced.ops.*

if nargin == 4
    % Alphaval not passed, (this should not happen in general)
    % MatlabFcn param is actually InputRanks
    % Alphaval param is actually MatlabFcn
    InputRanks = MatlabFcn;
    MatlabFcn = AlphaVal;
    AlphaVal = 1;
end

%'MatlabFcn' can be plus, minus, div, floor_divide, mul, or power
functionHandle = str2func(MatlabFcn);

%Inputs are already in reverse-pytorch
Xrank = InputRanks(1);
Yrank = InputRanks(2);

if ~isdlarray(Xval)
    Xval = single(Xval);
end

if ~isdlarray(Yval)
    Yval = single(Yval);
end

if MatlabFcn == "idivide"
    Xval = int64(floor(extractdata(Xval)));
    Yval = int64(floor(extractdata(Yval)));
end

% Function handle for floor_divide
if MatlabFcn == "@(x, y)floor(rdivide(x, y))" 
    isNeg = any(Xval < 0, 'all') || any(Yval < 0, 'all');
    % Warn the user that floor_divide operator can produce inference
    % mismatch in PyTorch 1.12 and earlier for negative valued tensors
    if isNeg
        warning(message('nnet_cnn_pytorchconverter:pytorchconverter:NumericalMismatchInOperatorForSpecifiedReason', ...
        "pyElementwiseBinary", "aten::floor_divide", "tensors with negative values"));
    end
end

% Warn the user if the power operator can produce complex number outputs
if MatlabFcn == "power"
    isFracExponent = any(Yval<1, 'all') && any(Yval>-1, 'all');
    isNegBase = any(Xval < 0, 'all');
    if isNegBase && isFracExponent
        warning(message('nnet_cnn_pytorchconverter:pytorchconverter:NumericalMismatchInOperatorForSpecifiedReason', ...
        "pyElementwiseBinary", "aten::pow", "tensors with negative values and fractional exponents"));
    end
end

% alpha is a scaling factor that belongs to the "aten::add" and "aten::sub" operators
if AlphaVal ~= 1
    Zval = functionHandle(Xval,Yval*single(AlphaVal));
else
    Zval = functionHandle(Xval,Yval);
end

%Get labels and reverse permutation from the max rank input

%When input ranks are equal, we get the output rank from
%the inputs
%Output will be in reverse PyTorch format

if Xrank == Yrank
    Zrank = Xrank;
else
    %When rank is not equal we get the rank from the input
    %with max rank. 
    [Zrank, ~] = max([Xrank, Yrank]);
end

Zval = dlarray(single(Zval),repmat('U',1,max(2, Zrank)));
end