function blobs = detectBlobsScaleFilter(im)
% DETECTBLOBS detects blobs in an image
%   BLOBS = DETECTBLOBSSCALEFILTER(IM, PARAM) detects multi-scale blobs in IM.
%   The method uses the Laplacian of Gaussian filter to find blobs across
%   scale space. This version of the code scales the filter and keeps the
%   image same which is slow for big filters.
% 
% Input:
%   IM - input image
%
% Ouput:
%   BLOBS - n x 4 array with blob in each row in (x, y, radius, score)
%
% This code is part of:
%
%   CMPSCI 670: Computer Vision, Fall 2014
%   University of Massachusetts, Amherst
%   Instructor: Subhransu Maji
%
%   Homework 3: Blob detector

n = 15;
init = 1.5;
thresh = 250;
k = 1.2;
[h, w] = size(rgb2gray(im));
scaleSpace = zeros(h,w,n);
f = @suppression;

for s = 1:n
    sig = init*k^s;
    height = 2*ceil(1.5*sig)+1; % returns first odd number greater than 3 sigma
    scaleSpace(:,:,s) = conv2(rgb2gray(im), s^2*fspecial('log',height,sig),'same').^2;
end

existence = false;
for s = 1:n
    if existence
        [l,~] = size(blobs);
    else
        l = 0;
    end
    base = scaleSpace(:,:,s);
    sig = init*k^s;
    oddRad = 2*ceil((sqrt(2)/2)*sig)+1;
    maxes = scaleSpace(:,:,1);
    for t = ceil((1+s)/2):ceil((s+n)/2)
        maxes = max(maxes, scaleSpace(:,:,t));
    end
    filter = nlfilter(maxes, [oddRad oddRad], f);
    scaleSpace(:,:,s) = base.*(base==filter);
    [rows,cols] = find(scaleSpace(:,:,s));
    if size(rows) >= 1
        for j = 1:size(rows)
            existence = true;
            blobs(j+l,:) = [cols(j),rows(j),sig,scaleSpace(rows(j),cols(j),s)];
        end
    end
end

    function y = suppression(x)
        m = max(max(x));
        [z,~] = size(x);
        center = ceil(z/2);
        if x(center,center) == m && m >= thresh
            y = m;
        else
            y = 0;
        end
    end
end