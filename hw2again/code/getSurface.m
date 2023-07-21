function  heightMap = getSurface(surfaceNormals, method)
% GETSURFACE computes the surface depth from normals
%   HEIGHTMAP = GETSURFACE(SURFACENORMALS, IMAGESIZE, METHOD) computes
%   HEIGHTMAP from the SURFACENORMALS using various METHODs. 
%  
% Input:
%   SURFACENORMALS: height x width x 3 array of unit surface normals
%   METHOD: the intergration method to be used
%
% Output:
%   HEIGHTMAP: height map of object

fx = -1*surfaceNormals(:,:,1)./surfaceNormals(:,:,3);
fy = -1*surfaceNormals(:,:,2)./surfaceNormals(:,:,3);
[h,w,~] = size(surfaceNormals);
heights = zeros(h,w);
cum_rows = cumsum(fy,1);
cum_cols = cumsum(fx,2);

switch method
    case 'column'
        for i = 1:h
            for j = 1:w
                heights(i,j) = cum_rows(i,1) + cum_cols(i,j);
            end
        end
        heightMap = heights;
    case 'row'
        for i = 1:h
            for j = 1:w
                heights(i,j) = cum_rows(i,j) + cum_cols(1,j);
            end
        end
        heightMap = heights;
    case 'average'
        heights2 = zeros(h,w);
        for i = 1:h
            for j = 1:w
                heights(i,j) = cum_rows(i,j) + cum_cols(1,j);
            end
        end
        for i = 1:h
            for j = 1:w
                heights2(i,j) = cum_rows(i,1) + cum_cols(i,j);
            end
        end
        heightMap = 0.5*(heights+heights2);
    case 'random'
        sum = 1;
        prep = zeros([h w sum]);
        for a = 1:sum
            for i = 1:h
                for j = 1:w
                    x = 1;
                    y = 1;
                    counter = 2;
                    path = zeros([1 2 1]);
                    path(:,:,1) = [1 1];
                    while x ~= j || y ~= i
                        if x > j
                            xpbias = 0.1;
                        else
                            xpbias = 0.4;
                        end
                        if y > i
                            ypbias = 0.1;
                        else
                            ypbias = 0.4;
                        end
                        random = rand;
                        if random < xpbias
                            if x == w
                                dx = -1;
                                dy = 0;
                            else
                                dx = 1;
                                dy = 0;
                            end
                        elseif random < 0.5
                            if x == 1
                                dx = 1;
                                dy = 0;
                            else
                                dx = -1;
                                dy = 0;
                            end
                        elseif random < 0.5 + ypbias
                            if y == h
                                dx = 0;
                                dy = -1;
                            else
                                dx = 0;
                                dy = 1;
                            end
                        else
                            if y == 1
                                dx = 0;
                                dy = 1;
                            else
                                dx = 0;
                                dy = -1;
                            end
                        end
                        x = x + dx;
                        y = y + dy;
                        path(:,:,counter) = [x,y];
                        counter = counter + 1;
                    end
                    [~,~,n] = size(path);
                    k = 2;
                    hor = [0 0];
                    vert = [0 0];
                    total = 0;
                    if i == 1 && j == 1
                        total = cum_rows(1,1) + cum_cols(1,1);
                    else
                        while k < n
                            if path(1,1,k) == path(1,1,k-1)
                                l = k;
                                while path(1,1,l) == path(1,1,k-1)
                                    l = l+1;
                                    if l > n
                                        break
                                    end
                                end
                                hor = reshape(path(1,:,l-1), [1 2]);
                                if k == 2
                                    total = total + cum_cols(1, hor(1));
                                else
                                    total = total + cum_cols(vert(2),hor(1)) - cum_cols(hor(2),hor(1));
                                end
                                k = l;
                            else
                                l = k;
                                while path(1,2,l) == path(1,2,k-1)
                                    l = l+1;
                                    if l > n
                                        break
                                    end
                                end
                                vert = reshape(path(1,:,l-1), [1 2]);
                                if k == 2
                                    total = total + cum_rows(vert(2), 1);
                                else
                                    total = total + cum_rows(vert(2),hor(1)) - cum_rows(vert(2),vert(1));
                                end
                                k = l;
                            end
                        end
                    end
                    prep(i,j,a) = total;
                end
            end
        end
        prep_sum = zeros([h w]);
        for b = 1:sum
            prep_sum = prep_sum + prep(:,:,b);
        end
        heightMap = (1/sum)*prep_sum;
end
