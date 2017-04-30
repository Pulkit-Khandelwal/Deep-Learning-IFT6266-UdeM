%This is a generalized implementation and might have to be tweaked to get
%the desired output
LABEL = imread('/Users/pulkit/Desktop/pulks/dcganL1_3.bmp'); % Ground Truth Image
SEGM = imread('/Users/pulkit/Desktop/pulks/dcganL1_true_3.bmp'); % Segmented Image
%LABEL = rgb2gray(LABEL);
%SEGM = rgb2gray(SEGM);
set1 = LABEL;    
set2 = SEGM;  
dicecoeff = 2*nnz(set1 & set2)/(nnz(set1) + nnz(set2))

imshow(LABEL)