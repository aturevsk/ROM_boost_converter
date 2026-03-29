function [Y, YRank] = pyLinear(X, W, B, inputRanks)
%PYLINEAR Applies a linear transformation to the input data.
% at::Tensor at::linear(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias = {})

% Copyright 2025 The MathWorks, Inc.

import neuralode_mlp_traced.ops.*

% Convert the input data to reverse-Python dimension order
X = stripdims(X);
W = stripdims(W);
if ~isempty(B)
    B = stripdims(B);
end

XRank = inputRanks(1);
BRank = inputRanks(2);
YRank = XRank;

% If X is a vector, ensure it is a column vector
if XRank==1
    X = [X(:)];
end

% If B is a vector, ensure it is a column vector
if BRank ==1
    B = [B(:)];
end

% The PyTorch input format for X
% (*, H_in), where * can be any number of dimensions, and H_in is the
% number of input features. W has format (H_out, H_in) and is transposed
% before being multipled by X. 
% In reverse Python, a 4-dimensional BCSS input will have format 
% (H_in, x3, x1, x2). 
% W will have shape (H_in, H_out). To multiply
% equivalent pages of X and W, pages of X must be transposed. 
Y = pagemtimes(X, 'transpose', W, 'none');

% After multiplication, pages of Yval are in reverse dimension order.
perm = 1:YRank;
perm(2) = 1;
perm(1) = 2;
Y = permute(Y, perm);

% Add bias 
if ~isempty(B)
    Y = Y + B;
end
end