function gtannotate(imgdir, resdir, varargin)
% usage: polyannotate(imgdir,resdir)
%  Annotate polygons in an image sequence.
%
% Input:
%  * imgdir: the image folder
%  * resdir: output folder
%  * options:
%       -nVmax: max vertice number
%       -th: distance threshold to define duplicated vertice
%       -start: the starting image index of the images list of imgdir
%       -ext: extension name for image files
% Output:
%  Annotations are saved to outfile with cell array gt, one entry per
%  frame.
%
% Writen by Xiaolu Sun and C. Mario Christoudias


options = struct(varargin{:});
if ~isfield(options, 'nVmax') 
    options.nVmax = 50;
end
if ~isfield(options, 'th') 
    options.th = 5;
end
if ~isfield(options, 'start') 
    options.start = 1;
end
if ~isfield(options, 'ext')
    options.ext = [];
end

if isempty(options.ext)
    imglist = dir([imgdir filesep '*.jpg']);
    if isempty(imglist)
        imglist = dir([imgdir filesep '*.JPG']);
    end
    if isempty(imglist)
        imglist = dir([imgdir filesep '*.JPEG']);
    end
else
    imglist = dir([imgdir filesep '*' options.ext]); 
end

if length(imglist)==0
    error('No images found!');
end
if ~exist(resdir, 'dir')
    mkdir(resdir);
end

fprintf('Key Instructions:\n\n');
fprintf('-----------load image---------\n');
fprintf('p: previous image\n');
fprintf('n: next image\n\n');
fprintf('-------annotate polygon-------\n');
fprintf('a/left-click: start annotating polygon\n');
fprintf('>>>d: quit current annotation without saving\n');
fprintf('>>>b: close the polygon by connecting the last vertex to the first one\n');
fprintf('>>>click the last vertex agian: delete the last vertex\n');
fprintf('>>>click the starting vertex agian: finish annotating current polygon\n');
fprintf('>>>otherwise: adding a new vertex\n\n');
fprintf('-------remove polygon---------\n');
fprintf('d: delete last polygon\n');
fprintf('h: delete the region clicked\n\n');
fprintf('-----save/load results--------\n');
fprintf('r: reload the saved results\n');
fprintf('s: save current gt \n');
fprintf('q: save and quit\n');
fprintf('f: quit without save\n\n');

done = 0;
fig = figure(1);
clf;
hold on;
idx = options.start;
n = length(imglist);
gt = [];

h = [];
[imgname,resname, gt] = showcurrent(imglist,imgdir,resdir,idx,gt);
ind = length(gt)+1;
while(~done)
    [x,y,key] = ginput(1);
    if isempty(key)
        continue;
    end
    switch key
        % previous image
        case 'p'
            savecurrent(resname,gt);
            idx = idx-1;
            idx = max(1,idx);
            gt = [];
            clf;
            [imgname,resname, gt] = showcurrent(imglist,imgdir,resdir,idx,gt);
            ind = length(gt)+1;
        % next image
        case 'n'
            savecurrent(resname,gt);
            idx = idx+1;
            idx = min(n,idx);
            gt = [];
            clf;
            [imgname,resname, gt] = showcurrent(imglist,imgdir,resdir,idx,gt);
            ind = length(gt)+1;
        % reload the saved results
        case 'r'
            gt = [];
            [imgname,resname, gt] = showcurrent(imglist,imgdir,resdir,idx,gt);
            ind = length(gt)+1;
        % save results
        case 's'
            savecurrent(resname,gt);
        % annotate
        case {'a',1} 
            hold on
            xs = [];
            ys = [];
%             if ~isempty(h), delete(h); end
            h = [];
            start = 1;
            ii = 1;
            flagadd = 1;
            while(ii < options.nVmax)
                [x,y,key] = ginput(1);
                if isempty(key)
                    continue;
                end
                switch(key)
                    % clear the current annotating polygon
                    case 'd'
                        flagadd = 0;
                        break;
                    % close the polygon by connecting the last vertex to
                    % the starting one
                    case 'b'
                        a = [xs; xs(1)];
                        b = [ys; ys(1)];
                        h = plot(a,b,'r.-', 'LineWidth', 2.0, 'MarkerSize', 8.0);
                        if length(a)<3
                            flagadd = 0;
                        end
                        break;
                    otherwise
                        info = imfinfo(imgname);
                        x = max(0,x);
                        x = min(x,info.Width);
                        y = max(0,y);
                        y = min(y,info.Height);
                        if ~isempty(h)
                            delete(h)
                            h  =[];
                        end
                        % remove last vertex
                        if ii>1&&norm([x-xs(end) y-ys(end)])<options.th
                            ii = ii-1;
                            xs(end) = [];
                            ys(end) = [];
                            if ~isempty(xs)
                                h = plot(xs,ys,'r.-', 'LineWidth', 2.0, 'MarkerSize', 8.0);
                            end
                            continue;
                        end
                        % finish the current annotation 
                        if ii>1&&norm([x-xs(1) y-ys(1)])<options.th
                            a = [xs; x];
                            b = [ys; y];
                            h = plot(a,b,'r.-', 'LineWidth', 2.0, 'MarkerSize', 8.0);
                            break;
                        end
                        xs = [xs; x];
                        ys = [ys; y];
                        h = plot(xs,ys,'r.-', 'LineWidth', 2.0, 'MarkerSize', 8.0);
                        ii = ii+1;
                end
            end
            if flagadd == 1
                gt{ind} = [xs ys];
                ind = ind+1;
            end
        % delete last polygon
        case 'd'
            if ~isempty(gt)
                if ind<=length(gt)
                    gt(ind:end) = [];
                end
                ind = ind-1;
                gt{ind} = [];
                [imgname,resname, gt] = showcurrent(imglist,imgdir,resdir,idx,gt);
            end
        % delete the polygon clicked by user
        case 'h'
            [x,y] = ginput(1);
            for i = 1:length(gt)
                if isempty(gt{i})
                    break;
                end
                if (inpolygon(x,y,gt{i}(:,1),gt{i}(:,2)))
                    gt(i) = [];
                    break;
                end
            end
            [imgname,resname, gt] = showcurrent(imglist,imgdir,resdir,idx,gt);
        % save and quit
        case 'q'
            savecurrent(resname,gt);
            done = 1;
        % quit without save
        case 'f'
%             savecurrent(resname,gt);
            done = 1;
    end
end

close(fig);

end
function [imgname,resname, gt] = showcurrent(imglist,imgdir,resdir,idx,gt)
[pathstr,name,ext] = fileparts(imglist(idx).name);
imgname = fullfile(imgdir, imglist(idx).name);
resname = fullfile(resdir, [name '.mat']);
n = length(imglist);
title(sprintf('Image %d/%d: %s', idx, n, imgname));
imshow(imgname);
hold on;
if ~isempty(gt)
    for i = 1:length(gt)
        if isempty(gt{i})
            gt(i) = [];
            break;
        end
        ee = [gt{i}(:,:); gt{i}(1,:)];
        h = plot(ee(:,1),ee(:,2),'r.-', 'LineWidth', 2.0, 'MarkerSize', 8.0);
    end
    return;
end
if exist(resname,'file')
    load(resname);
    dellist = [];
    for i = 1:length(gt)
        if isempty(gt{i})
            dellist = [dellist i];
        end
    end
    gt(dellist) = [];
    for i = 1:length(gt)
        ee = [gt{i}(:,:); gt{i}(1,:)];
        h = plot(ee(:,1),ee(:,2),'r.-', 'LineWidth', 2.0, 'MarkerSize', 8.0);
    end
end
end
function savecurrent(resname,gt)
if ~isempty(gt)
    if isempty(gt{end})
        gt(end) = [];
    end
    save(resname, 'gt');
end
end
