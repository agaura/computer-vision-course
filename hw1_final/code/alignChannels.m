function [imShift, predShift] = alignChannels(im, maxShift)
% ALIGNCHANNELS align channels in an image.
%   [IMSHIFT, PREDSHIFT] = ALIGNCHANNELS(IM, MAXSHIFT) aligns the channels in an
%   NxMx3 image IM. The first channel is fixed and the remaining channels
%   are aligned to it within the maximum displacement range of MAXSHIFT (in
%   both directions). The code returns the aligned image IMSHIFT after
%   performing this alignment. The optimal shifts are returned as in
%   PREDSHIFT a 2x2 array. PREDSHIFT(1,:) is the shifts  in I (the first) 
%   and J (the second) dimension of the second channel, and PREDSHIFT(2,:)
%   are the same for the third channel.


% Sanity check
assert(size(im,3) == 3);
assert(all(maxShift > 0));

original = im(:,:,1);

for k = -maxShift(1):maxShift(1)
    for l = -maxShift(2):maxShift(2)
        exp1 = circshift(im(:,:,2), [k l]);
        exp2 = circshift(im(:,:,3), [k l]);
        nExp1 = exp1(:)/norm(exp1(:));
        nExp2 = exp2(:)/norm(exp2(:));
        comp1(k+maxShift(1)+1,l+maxShift(2)+1) = dot(nExp1,original(:));
        comp2(k+maxShift(1)+1,l+maxShift(2)+1) = dot(nExp2,original(:));
    end
end

[~, indices] = max(comp1(:));
[minI1, minJ1] = ind2sub([size(comp1,1) size(comp1,2)],indices);
minI1 = minI1 - maxShift(1) - 1;
minJ1 = minJ1 - maxShift(2) - 1;
[~, indices] = max(comp2(:));
[minI2, minJ2] = ind2sub([size(comp2,1) size(comp2,2)],indices);
minI2 = minI2 - maxShift(1) - 1;
minJ2 = minJ2 - maxShift(2) - 1;

predShift = [minI1 minJ1; minI2 minJ2];
im(:,:,2) = circshift(im(:,:,2), [minI1 minJ1]);
im(:,:,3) = circshift(im(:,:,3), [minI2 minJ2]);
imShift = im;