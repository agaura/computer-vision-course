function output = demosaicImage(im, method)
% DEMOSAICIMAGE computes the color image from mosaiced input
%   OUTPUT = DEMOSAICIMAGE(IM, METHOD) computes a demosaiced OUTPUT from
%   the input IM. The choice of the interpolation METHOD can be 
%   'baseline', 'nn', 'linear', 'adagrad'. 

switch lower(method)
    case 'baseline'
        output = demosaicBaseline(im);
    case 'nn'
        output = demosaicNN(im);         
    case 'linear'
        output = demosaicLinear(im);     
    case 'adagrad'
        output = demosaicAdagrad(im);    
end

%--------------------------------------------------------------------------
%                          Baseline demosaicing algorithm. 
%                          The algorithm replaces missing values with the
%                          mean of each color channel.
%--------------------------------------------------------------------------
function mosim = demosaicBaseline(im)
mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
[imageHeight, imageWidth] = size(im);

% Red channel (odd rows and columns);
redValues = im(1:2:imageHeight, 1:2:imageWidth);
meanValue = mean(mean(redValues));
mosim(:,:,1) = meanValue;
mosim(1:2:imageHeight, 1:2:imageWidth,1) = im(1:2:imageHeight, 1:2:imageWidth);

% Blue channel (even rows and colums);
blueValues = im(2:2:imageHeight, 2:2:imageWidth);
meanValue = mean(mean(blueValues));
mosim(:,:,3) = meanValue;
mosim(2:2:imageHeight, 2:2:imageWidth,3) = im(2:2:imageHeight, 2:2:imageWidth);

% Green channel (remaining places)
% We will first create a mask for the green pixels (+1 green, -1 not green)
mask = ones(imageHeight, imageWidth);
mask(1:2:imageHeight, 1:2:imageWidth) = -1;
mask(2:2:imageHeight, 2:2:imageWidth) = -1;
greenValues = mosim(mask > 0);
meanValue = mean(greenValues);
% For the green pixels we copy the value
greenChannel = im;
greenChannel(mask < 0) = meanValue;
mosim(:,:,2) = greenChannel;

%--------------------------------------------------------------------------
%                           Nearest neighbour algorithm
%--------------------------------------------------------------------------
function mosim = demosaicNN(im)
mosim = repmat(im, [1 1 3]);
[imageHeight, imageWidth] = size(im);

adjI = 1;
if rem(imageHeight,2) == 0 
    adjI = 0;
end
adjJ = 1;
if rem(imageWidth,2) == 0 
    adjJ = 0;
end

% give all reds in a 2x2 the top-left corner value
mosim(1:2:imageHeight, 1:2:imageWidth,1) = im(1:2:imageHeight, 1:2:imageWidth);
mosim(1:2:imageHeight, 2:2:imageWidth,1) = im(1:2:imageHeight, 1:2:imageWidth-adjJ);
mosim(2:2:imageHeight, 1:2:imageWidth,1) = im(1:2:imageHeight-adjI, 1:2:imageWidth);
mosim(2:2:imageHeight, 2:2:imageWidth,1) = im(1:2:imageHeight-adjI, 1:2:imageWidth-adjJ);

% give all blues in a 2x2 the bottom-right corner value
mosim(1:2:imageHeight-adjI, 1:2:imageWidth-adjJ,3) = im(2:2:imageHeight, 2:2:imageWidth);
mosim(1:2:imageHeight-adjI, 2:2:imageWidth-adjJ,3) = im(2:2:imageHeight, 2:2:imageWidth);
mosim(2:2:imageHeight-adjI, 1:2:imageWidth-adjJ,3) = im(2:2:imageHeight, 2:2:imageWidth);
mosim(2:2:imageHeight-adjI, 2:2:imageWidth-adjJ,3) = im(2:2:imageHeight, 2:2:imageWidth);
% the last row, if there is one, takes the top-right value
if rem(imageHeight,2) == 1
    mosim(imageHeight, 1:2:imageWidth-adjJ,3) = im(imageHeight-1, 2:2:imageWidth);
    mosim(imageHeight, 2:2:imageWidth-adjJ,3) = im(imageHeight-1, 2:2:imageWidth);
end
% the last column, if there is one, takes the bottom-left value
if rem(imageWidth,2) == 1
    mosim(1:2:imageHeight-adjI, imageWidth,3) = im(2:2:imageHeight, imageWidth-1);
    mosim(2:2:imageHeight-adjI, imageWidth,3) = im(2:2:imageHeight, imageWidth-1);
end
% corner takes diagonal
if rem(imageHeight,2) == 1 && rem(imageWidth,2) == 1
    mosim(imageHeight, imageWidth) = mosim(imageHeight-1, imageWidth-1);
end

% give odd rowed greens the right value, even rows the left value
mosim(1:2:imageHeight, 1:2:imageWidth-adjJ,2) = im(1:2:imageHeight, 2:2:imageWidth);
mosim(1:2:imageHeight, 2:2:imageWidth,2) = im(1:2:imageHeight, 2:2:imageWidth);
mosim(2:2:imageHeight, 1:2:imageWidth,2) = im(2:2:imageHeight, 1:2:imageWidth);
mosim(2:2:imageHeight, 2:2:imageWidth,2) = im(2:2:imageHeight, 1:2:imageWidth-adjJ);
% the last column's greens all take the left value if there is odd width
if rem(imageWidth,2) == 1
    mosim(1:2:imageHeight, imageWidth,2) = im(1:2:imageHeight, imageWidth-1);
end

%--------------------------------------------------------------------------
%                           Linear interpolation
%--------------------------------------------------------------------------
function mosim = demosaicLinear(im)
mosim = repmat(im, [1 1 3]);
[imageHeight, imageWidth] = size(im);

% reds with value
mosim(1:2:imageHeight, 1:2:imageWidth,1) = im(1:2:imageHeight, 1:2:imageWidth);
% reds surrounded left and right by reds
mosim(1:2:imageHeight, 2:2:imageWidth-1,1) = 0.5*(im(1:2:imageHeight, ...
    1:2:imageWidth-2) + im(1:2:imageHeight, 3:2:imageWidth));
% reds surrounded up and down by reds
mosim(2:2:imageHeight-1, 1:2:imageWidth,1) = 0.5*(im(1:2:imageHeight-2, ...
    1:2:imageWidth) + im(3:2:imageHeight, 1:2:imageWidth));
% reds surrounded on all sides by reds
mosim(2:2:imageHeight-1, 2:2:imageWidth-1,1) = 0.25*(im(1:2:imageHeight-2, ...
    1:2:imageWidth-2) + im(1:2:imageHeight-2, 3:2:imageWidth) + im(3:2:imageHeight, ...
    1:2:imageWidth-2) + im(3:2:imageHeight, 3:2:imageWidth));
% edges
if rem(imageHeight,2) == 0
    mosim(imageHeight, 1:2:imageWidth,1) = im(imageHeight-1, 1:2:imageWidth,1);
    mosim(imageHeight, 2:2:imageWidth-1,1) = 0.5*(im(imageHeight-1, 1:2:imageWidth-2,1) + ...
        im(imageHeight-1, 3:2:imageWidth,1));
end
if rem(imageWidth,2) == 0
    mosim(1:2:imageHeight, imageWidth,1) = im(1:2:imageHeight, imageWidth-1,1);
    mosim(2:2:imageHeight-1, imageWidth,1) = 0.5*(im(1:2:imageHeight-2, imageWidth-1,1) + ...
        im(3:2:imageHeight, imageWidth-1,1));
end
if rem(imageHeight,2) == 0 && rem(imageWidth,2) == 0
    mosim(imageHeight, imageWidth,1) = im(imageHeight-1, imageWidth-1,1);
end

% blues with value
mosim(2:2:imageHeight, 2:2:imageWidth,3) = im(2:2:imageHeight, 2:2:imageWidth);
% blues surrounded left and right by blues
mosim(2:2:imageHeight, 3:2:imageWidth-1,3) = 0.5*(im(2:2:imageHeight, ...
    2:2:imageWidth-2) + im(2:2:imageHeight, 4:2:imageWidth));
% blues surrounded up and down by blues
mosim(3:2:imageHeight-1, 2:2:imageWidth,3) = 0.5*(im(2:2:imageHeight-2, ...
    2:2:imageWidth) + im(4:2:imageHeight, 2:2:imageWidth));
% blues surrounded on all sides by blues
mosim(3:2:imageHeight-1, 3:2:imageWidth-1,3) = 0.25*(im(2:2:imageHeight-2, ...
    2:2:imageWidth-2) + im(2:2:imageHeight-2, 4:2:imageWidth) + im(4:2:imageHeight, ...
    2:2:imageWidth-2) + im(4:2:imageHeight, 4:2:imageWidth));
% edges
mosim(1, 1,3) = im(2,2);
mosim(1, 2:2:imageWidth,3) = im(2, 2:2:imageWidth);
mosim(1, 3:2:imageWidth-1,3) = 0.5*(im(2, 2:2:imageWidth-2) + im(2, 4:2:imageWidth));
mosim(2:2:imageHeight, 1,3) = im(2:2:imageHeight, 2);
mosim(3:2:imageHeight-1, 1,3) = 0.5*(im(2:2:imageHeight-2, 2) + im(4:2:imageHeight, 2));
if rem(imageHeight,2) == 1
    mosim(1, imageWidth,3) = im(2, imageWidth-1);
    mosim(2:2:imageHeight, imageWidth,3) = im(2:2:imageHeight, imageWidth-1);
    mosim(3:2:imageHeight-1, imageWidth,3) = 0.5*(im(2:2:imageHeight-2, imageWidth-1) + ...
        im(4:2:imageHeight, imageWidth-1));
end
if rem(imageWidth,2) == 1
    mosim(imageHeight, 1,3) = im(imageHeight-1, 2);
    mosim(imageHeight, 2:2:imageWidth,3) = im(imageHeight-1, 2:2:imageWidth);
    mosim(imageHeight, 3:2:imageWidth-1,3) = 0.5*(im(imageHeight-1, 2:2:imageWidth-2) + ...
        im(imageHeight-1, 4:2:imageWidth));
end
if rem(imageHeight,2) == 1 && rem(imageWidth,2) == 1
    mosim(imageHeight, imageWidth,3) = im(imageHeight-1, imageWidth-1,1);
end

% greens with value
mosim(1:2:imageHeight, 2:2:imageWidth,2) = im(1:2:imageHeight, 2:2:imageWidth);
mosim(2:2:imageHeight, 1:2:imageWidth,2) = im(2:2:imageHeight, 1:2:imageWidth);
% greens surrounded on all sides by greens
mosim(2:2:imageHeight-1, 2:2:imageWidth-1,2) = 0.25*(im(1:2:imageHeight-2, ...
    2:2:imageWidth-1) + im(3:2:imageHeight, 2:2:imageWidth-1) + im(2:2:imageHeight-1, ...
    1:2:imageWidth-2) + im(2:2:imageHeight-1, 3:2:imageWidth));
mosim(3:2:imageHeight-1, 3:2:imageWidth-1,2) = 0.25*(im(2:2:imageHeight-2, ...
    3:2:imageWidth-1) + im(4:2:imageHeight, 3:2:imageWidth-1) + im(3:2:imageHeight-1, ...
    2:2:imageWidth-2) + im(3:2:imageHeight-1, 4:2:imageWidth));
% edges
mosim(1, 1,2) = 0.5*(im(1, 2) + im(2, 1));
mosim(1, 3:2:imageWidth-1,2) = (1/3)*(im(1, 2:2:imageWidth-2) + ...
    im(1, 4:2:imageWidth) + im(2, 3:2:imageWidth-1));
mosim(3:2:imageHeight-1, 1,2) = (1/3)*(im(2:2:imageHeight-2, 1) + ...
    im(4:2:imageHeight, 1) + im(3:2:imageHeight-1, 2));
if rem(imageHeight,2) == 0
    mosim(imageHeight, 2:2:imageWidth-1,2) = (1/3)*(im(imageHeight, 1:2:imageWidth-2) + ...
        im(imageHeight, 3:2:imageWidth) + im(imageHeight-1, 2:2:imageWidth-1));
end
if rem(imageHeight,2) == 1
    mosim(imageHeight, 1,2) = 0.5*(im(imageHeight-1, 1) + im(imageHeight, 2));
    mosim(imageHeight, 3:2:imageWidth-1,2) = (1/3)*(im(imageHeight, 2:2:imageWidth-2) + ...
        im(imageHeight, 4:2:imageWidth) + im(imageHeight-1, 3:2:imageWidth-1));
end
if rem(imageWidth,2) == 0
    mosim(2:2:imageHeight-1, imageWidth,2) = (1/3)*(im(1:2:imageHeight-2, imageWidth) + ...
        im(3:2:imageHeight, imageWidth) + im(2:2:imageHeight-1, imageWidth-1)); 
end
if rem(imageWidth,2) == 1
    mosim(1, imageWidth,2) = 0.5*(im(1, imageWidth-1) + im(2, imageWidth));
    mosim(3:2:imageHeight-1, imageWidth,2) = (1/3)*(im(2:2:imageHeight-2, imageWidth) + ...
        im(4:2:imageHeight, imageWidth) + im(3:2:imageHeight-1, imageWidth-1)); 
end
if (rem(imageHeight, 2) ~= rem(imageWidth, 2))
    mosim(imageHeight, imageWidth,2) = 0.5*(im(imageHeight-1,imageWidth) + ...
        im(imageHeight, imageWidth-1));
end

%--------------------------------------------------------------------------
%                           Adaptive gradient
%--------------------------------------------------------------------------
function mosim = demosaicAdagrad(im)

mosim = repmat(im, [1 1 3]);
[imageHeight, imageWidth] = size(im);

% reds with value
mosim(1:2:imageHeight, 1:2:imageWidth,1) = im(1:2:imageHeight, 1:2:imageWidth);
% reds surrounded left and right by reds
mosim(1:2:imageHeight, 2:2:imageWidth-1,1) = 0.5*(im(1:2:imageHeight, ...
    1:2:imageWidth-2) + im(1:2:imageHeight, 3:2:imageWidth));
% reds surrounded up and down by reds
mosim(2:2:imageHeight-1, 1:2:imageWidth,1) = 0.5*(im(1:2:imageHeight-2, ...
    1:2:imageWidth) + im(3:2:imageHeight, 1:2:imageWidth));
% reds surrounded on all sides by reds with diagonal gradients
for k = 2:2:imageHeight-1
    for l = 2:2:imageWidth-1
        if abs(im(k-1, l-1) - im(k+1, l+1)) <= abs(im(k+1, l-1) - im(k-1, l+1))
            mosim(k, l,1) = 0.5*(im(k-1, l-1) + im(k+1, l+1));
        else
            mosim(k, l,1) = 0.5*(im(k+1, l-1) + im(k-1, l+1));
        end
    end
end
% edges
if rem(imageHeight,2) == 0
    mosim(imageHeight, 1:2:imageWidth,1) = im(imageHeight-1, 1:2:imageWidth,1);
    mosim(imageHeight, 2:2:imageWidth-1,1) = 0.5*(im(imageHeight-1, 1:2:imageWidth-2,1) + ...
        im(imageHeight-1, 3:2:imageWidth,1));
end
if rem(imageWidth,2) == 0
    mosim(1:2:imageHeight, imageWidth,1) = im(1:2:imageHeight, imageWidth-1,1);
    mosim(2:2:imageHeight-1, imageWidth,1) = 0.5*(im(1:2:imageHeight-2, imageWidth-1,1) + ...
        im(3:2:imageHeight, imageWidth-1,1));
end
if rem(imageHeight,2) == 0 && rem(imageWidth,2) == 0
    mosim(imageHeight, imageWidth,1) = im(imageHeight-1, imageWidth-1,1);
end

% blues with value
mosim(2:2:imageHeight, 2:2:imageWidth,3) = im(2:2:imageHeight, 2:2:imageWidth);
% blues surrounded left and right by blues
mosim(2:2:imageHeight, 3:2:imageWidth-1,3) = 0.5*(im(2:2:imageHeight, ...
    2:2:imageWidth-2) + im(2:2:imageHeight, 4:2:imageWidth));
% blues surrounded up and down by blues
mosim(3:2:imageHeight-1, 2:2:imageWidth,3) = 0.5*(im(2:2:imageHeight-2, ...
    2:2:imageWidth) + im(4:2:imageHeight, 2:2:imageWidth));
% blues surrounded on all sides by blues with diagonal gradients
for k = 3:2:imageHeight-1
    for l = 3:2:imageWidth-1
        if abs(im(k-1, l-1) - im(k+1, l+1)) <= abs(im(k+1, l-1) - im(k-1, l+1))
            mosim(k, l,3) = 0.5*(im(k-1, l-1) + im(k+1, l+1));
        else
            mosim(k, l,3) = 0.5*(im(k+1, l-1) + im(k-1, l+1));
        end
    end
end
% edges
mosim(1, 1,3) = im(2,2);
mosim(1, 2:2:imageWidth,3) = im(2, 2:2:imageWidth);
mosim(1, 3:2:imageWidth-1,3) = 0.5*(im(2, 2:2:imageWidth-2) + im(2, 4:2:imageWidth));
mosim(2:2:imageHeight, 1,3) = im(2:2:imageHeight, 2);
mosim(3:2:imageHeight-1, 1,3) = 0.5*(im(2:2:imageHeight-2, 2) + im(4:2:imageHeight, 2));
if rem(imageHeight,2) == 1
    mosim(1, imageWidth,3) = im(2, imageWidth-1);
    mosim(2:2:imageHeight, imageWidth,3) = im(2:2:imageHeight, imageWidth-1);
    mosim(3:2:imageHeight-1, imageWidth,3) = 0.5*(im(2:2:imageHeight-2, imageWidth-1) + ...
        im(4:2:imageHeight, imageWidth-1));
end
if rem(imageWidth,2) == 1
    mosim(imageHeight, 1,3) = im(imageHeight-1, 2);
    mosim(imageHeight, 2:2:imageWidth,3) = im(imageHeight-1, 2:2:imageWidth);
    mosim(imageHeight, 3:2:imageWidth-1,3) = 0.5*(im(imageHeight-1, 2:2:imageWidth-2) + ...
        im(imageHeight-1, 4:2:imageWidth));
end
if rem(imageHeight,2) == 1 && rem(imageWidth,2) == 1
    mosim(imageHeight, imageWidth,3) = im(imageHeight-1, imageWidth-1,1);
end

% greens with value
mosim(1:2:imageHeight, 2:2:imageWidth,2) = im(1:2:imageHeight, 2:2:imageWidth);
mosim(2:2:imageHeight, 1:2:imageWidth,2) = im(2:2:imageHeight, 1:2:imageWidth);
% greens surrounded on all sides by greens
for k = 2:2:imageHeight-1
    for l = 2:2:imageWidth-1
        if abs(im(k, l-1) - im(k, l+1)) <= abs(im(k-1, l) - im(k+1, l))
            mosim(k, l,2) = 0.5*(im(k, l-1) + im(k, l+1));
        else
            mosim(k, l,2) = 0.5*(im(k-1, l) + im(k+1, l));
        end
    end
end
for k = 3:2:imageHeight-1
    for l = 3:2:imageWidth-1
        if abs(im(k, l-1) - im(k, l+1)) <= abs(im(k-1, l) - im(k+1, l))
            mosim(k, l,2) = 0.5*(im(k, l-1) + im(k, l+1));
        else
            mosim(k, l,2) = 0.5*(im(k-1, l) + im(k+1, l));
        end
    end
end
% edges
mosim(1, 1,2) = 0.5*(im(1, 2) + im(2, 1));
mosim(1, 3:2:imageWidth-1,2) = (1/3)*(im(1, 2:2:imageWidth-2) + ...
    im(1, 4:2:imageWidth) + im(2, 3:2:imageWidth-1));
mosim(3:2:imageHeight-1, 1,2) = (1/3)*(im(2:2:imageHeight-2, 1) + ...
    im(4:2:imageHeight, 1) + im(3:2:imageHeight-1, 2));
if rem(imageHeight,2) == 0
    mosim(imageHeight, 2:2:imageWidth-1,2) = (1/3)*(im(imageHeight, 1:2:imageWidth-2) + ...
        im(imageHeight, 3:2:imageWidth) + im(imageHeight-1, 2:2:imageWidth-1));
end
if rem(imageHeight,2) == 1
    mosim(imageHeight, 1,2) = 0.5*(im(imageHeight-1, 1) + im(imageHeight, 2));
    mosim(imageHeight, 3:2:imageWidth-1,2) = (1/3)*(im(imageHeight, 2:2:imageWidth-2) + ...
        im(imageHeight, 4:2:imageWidth) + im(imageHeight-1, 3:2:imageWidth-1));
end
if rem(imageWidth,2) == 0
    mosim(2:2:imageHeight-1, imageWidth,2) = (1/3)*(im(1:2:imageHeight-2, imageWidth) + ...
        im(3:2:imageHeight, imageWidth) + im(2:2:imageHeight-1, imageWidth-1)); 
end
if rem(imageWidth,2) == 1
    mosim(1, imageWidth,2) = 0.5*(im(1, imageWidth-1) + im(2, imageWidth));
    mosim(3:2:imageHeight-1, imageWidth,2) = (1/3)*(im(2:2:imageHeight-2, imageWidth) + ...
        im(4:2:imageHeight, imageWidth) + im(3:2:imageHeight-1, imageWidth-1)); 
end
if (rem(imageHeight, 2) ~= rem(imageWidth, 2))
    mosim(imageHeight, imageWidth,2) = 0.5*(im(imageHeight-1,imageWidth) + ...
        im(imageHeight, imageWidth-1));
end
