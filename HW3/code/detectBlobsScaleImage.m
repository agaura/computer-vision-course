function blobs = detectBlobsScaleImage(im)
% DETECTBLOBS detects blobs in an image
%   BLOBS = DETECTBLOBSCALEIMAGE(IM, PARAM) detects multi-scale blobs in IM.
%   The method uses the Laplacian of Gaussian filter to find blobs across
%   scale space. This version of the code scales the image and keeps the
%   filter same for speed. 
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

n = 30;
init = 1.75;
thresh = 1000;
k = 1.05;
[h, w] = size(rgb2gray(im));
scaleSpace = zeros(h,w,n);
f = @suppression;

for s = 1:n
    sig = init*k;
    height = 2*ceil(1.5*sig)+1; % returns first odd number greater than 3 sigma
    res = imresize(rgb2gray(im),1/k^s);
    scaleSpace(:,:,s) = imresize(conv2(res, s^2*fspecial('log',height,sig),'same').^2,[h w]);
end

existence = false;
for s = 1:n
    if existence
        [l,~] = size(blobs);
    else
        l = 0;
    end
    oddRad = 3;
    idealBase = imresize(scaleSpace(:,:,s),1/k^s);
    [nh, nw] = size(idealBase);
    totals = zeros(nh,nw);
    for i = 0:3
        base = imresize(rot90(scaleSpace(:,:,s),i),1/k^s,'nearest');
        [nh, nw] = size(base);
        idealBase = max(idealBase, rot90(base,4-i));
        if s == 1
            maxes = max(imresize(rot90(scaleSpace(:,:,s),i),[nh nw],'nearest'),...
            imresize(rot90(scaleSpace(:,:,s+1),i),[nh nw],'nearest'));
        elseif s == n
            maxes = max(imresize(rot90(scaleSpace(:,:,s),i),[nh nw],'nearest'),...
            imresize(rot90(scaleSpace(:,:,s-1),i),[nh nw],'nearest'));
        else
            maxes = max(imresize(rot90(scaleSpace(:,:,s),i),[nh nw],'nearest'),...
            max(imresize(rot90(scaleSpace(:,:,s+1),i),[nh nw],'nearest'),...
            imresize(rot90(scaleSpace(:,:,s-1),i),[nh nw],'nearest')));
        end
        totals = max(totals, rot90(maxes, 4-i));
    end
    filter = nlfilter(totals, [oddRad oddRad], f);
    output = idealBase.*(idealBase==filter);
    [nh, nw] = size(imresize(scaleSpace(:,:,s),1/k^s));
    [rows,cols] = find(output);
    if size(rows) >= 1
        for j = 1:size(rows)
            existence = true;
            blobs(j+l,:) = [ceil(cols(j)*w/nw),ceil(rows(j)*h/nh),init*k^s,output(rows(j),cols(j))];
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