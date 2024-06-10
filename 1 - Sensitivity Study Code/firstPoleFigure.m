clear all
close all
clc

%your_MTEX_path = '/Users/moorej/texture/mtex-5.1.1/'
%addpath your_MTEX_path
%startup_mtex

%cd /Users/moorej/texture/mtex-5.1.1/
%startup_mtex

crystalType = 'cubic';
%crystalType = 'hcp';

% Input
%fname = ['texture.txt'];
%fname = ['textureRnd.txt'];
%fname = ['textureRnd3.txt'];
%fname = ['textureTi64100.txt'];
%fname = ['textureTi64100pct59.txt'];
fname = ['texture_new.txt'];

%Output of Euler Angles
%outName = ['textureRnd3Euler.txt']
%outName = ['textureTi64100Euler.txt']
%outName = ['textureTi64100pct59Euler.txt'];
outName = ['textureEuler_new.txt'];

switch lower(crystalType)
    case 'cubic'
        cs = crystalSymmetry('cubic')
        
        % I did not have this before
        %ss = specimenSymmetry('cubic');
        ss = specimenSymmetry('triclinic');

        quat = load(fname);
        eul = quat2eul(quat,'zyz');

        [o w] = loadOrientation_generic(eul,'CS',cs,'SS',ss, 'ColumnNames', ...
        {'phi2' 'Phi' 'phi1' },'Columns',[1,2,3],'Radians','Matthies');

        %plotPDF(o,Miller({1,0,0},{1,1,1},{1,1,0},cs),'all');
        plotPDF(o,Miller({1,0,0},{1,1,1},{1,1,0},cs),'contourf','all');

    
    case 'hcp'
        cs = crystalSymmetry('6/mmm');
        quat = load(fname);
        eul = quat2eul(quat,'zyz');
        
        % I did not have this before
        ss = specimenSymmetry('triclinic');
        
        [o w] = loadOrientation_generic(eul,'CS',cs,'SS',ss, 'ColumnNames', ...
            {'phi2' 'Phi' 'phi1' },'Columns',[1,2,3],'Radians','Matthies');
        
        %[o w] = loadOrientation_generic(eul,'CS',cs)
        
        %plotPDF(o,Miller({0,0,0,1},{1,1,-2,0},{1,-1,0,0},cs),'all');
        plotPDF(o,Miller({0,0,0,1},{1,1,-2,0},{1,-1,0,0},cs),'contourf','all');
end

%write euler angles to a file 
dlmwrite(outName, rad2deg(eul));